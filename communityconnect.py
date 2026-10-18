import click
import base64
import hashlib
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from flask import Flask, abort, flash, g, has_request_context, jsonify, redirect, render_template, request, session, url_for
from functools import lru_cache, wraps
from core.ai_client import ask_ai, embed_texts, get_backend_status
from core.ai_schemas import (
    ALLOWED_CATEGORIES,
    ALLOWED_LANGUAGES,
    ALLOWED_URGENCY,
    TRANSLATION_LANGUAGES,
    validate_document_summary,
    validate_intake,
    validate_translation,
    validate_triage,
)
from core.embedding_cache import load_or_create_embeddings_with_backend
from core.language_detection import detect_user_language
from core.place_catalog import load_place_catalog, place_options
from core.privacy import deidentify_text, restore_redactions
from core.scoring import calculate_match, cosine_similarity, eligibility_reasons, rank_matches
from html import unescape
from supabase import create_client
import hmac
import json
import math
import os
import re
import secrets
from threading import Lock
from time import monotonic
import unicodedata
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from werkzeug.exceptions import HTTPException

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("COMMUNITYCONNECT_SECRET_KEY")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("COMMUNITYCONNECT_HTTPS_ONLY") == "1" or os.environ.get("VERCEL") == "1"
)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

ALLOWED_ROLES = {"seeker", "helper", "volunteer", "ngo"}
HELPER_ROLES = {"helper", "volunteer", "ngo"}
DEMO_PASSWORD = os.environ.get("COMMUNITYCONNECT_DEMO_PASSWORD", "")
EMBEDDING_MODEL_ID = os.environ.get(
    "AI_EMBEDDING_MODEL_ID",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_MODEL_VERSION = os.environ.get("AI_EMBEDDING_MODEL_VERSION", "v2")
REQUEST_CATEGORIES = sorted(ALLOWED_CATEGORIES)
TRANSLATION_LANGUAGE_NAMES = {"it": "Italian", "en": "English", "es": "Spanish", "fr": "French"}
SPOKEN_LANGUAGE_OPTIONS = {"it": "Italian", "en": "English", "es": "Spanish", "fr": "French"}
LANGUAGE_PREFERENCE_COOKIE = "communityconnect_language"
_TRANSLATION_BACKOFF_SECONDS = 300
_TRANSLATION_REQUEST_BUDGET_SECONDS = 0.75
_translation_backoff_until = {}
_translation_backoff_lock = Lock()
AVAILABILITY_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
AVAILABILITY_PARTS = ["morning", "afternoon", "evening"]
EMERGENCY_PATTERN = re.compile(
    r"\b(?:112|can't breathe|cannot breathe|difficulty breathing|chest pain|severe bleeding|"
    r"unconscious|stroke|heart attack|overdose|suicid(?:e|al)|self[- ]harm|"
    r"non riesco a respirare|difficolt[aà] respiratorie|dolore al petto|emorragia|"
    r"incosciente|ictus|infarto|overdose|suicid\w*|autolesion\w*)\b",
    re.IGNORECASE,
)
MEDICAL_ADVICE_PATTERN = re.compile(
    r"\b(?:medical advice|diagnos\w*|what (?:medicine|medication|dose) should i|"
    r"which (?:medicine|medication)|dosage|what should i take|consiglio medico|"
    r"diagnos\w*|che farmaco|quale farmaco|quale dose|dosaggio|quale terapia)\b",
    re.IGNORECASE,
)
LEGAL_ADVICE_PATTERN = re.compile(
    r"\b(?:legal advice|lawyer|solicitor|sue (?:them|my|the)|interpret (?:this|my) contract|"
    r"consulenza legale|avvocato|fare causa|interpretare (?:questo|il) contratto|"
    r"quali sono i miei diritti)\b",
    re.IGNORECASE,
)

CATEGORIES = {
    "math": "Education",
    "maths": "Education",
    "matematica": "Education",
    "ripetizioni": "Education",
    "tutor": "Education",
    "tutoring": "Education",
    "tuition": "Education",
    "teach": "Education",
    "computer": "Digital Support",
    "online": "Digital Support",
    "technology": "Digital Support",
    "digitale": "Digital Support",
    "spid": "Digital Support",
    "moduli": "Digital Support",
    "food": "Food Support",
    "groceries": "Food Support",
    "grocery": "Food Support",
    "spesa": "Food Support",
    "commissioni": "Community Support",
    "medicine": "Healthcare",
    "health": "Healthcare",
    "elderly": "Care Support",
    "companionship": "Care Support",
    "caregiving": "Care Support",
    "anziani": "Care Support",
    "accompagnamento": "Care Support",
    "affitto": "Housing Support",
    "affitti": "Housing Support",
    "housing": "Housing Support",
    "lease": "Housing Support",
    "rental": "Housing Support",
    "rentals": "Housing Support",
    "rent": "Housing Support",
    "room": "Housing Support",
    "rooms": "Housing Support",
    "stanza": "Housing Support",
    "stanze": "Housing Support",
    "apartment": "Housing Support",
    "apartments": "Housing Support",
    "accommodation": "Housing Support",
    "accommodations": "Housing Support",
    "casa": "Housing Support",
    "traduzione": "Language Support",
    "interpreter": "Language Support",
    "interpretation": "Language Support",
    "translation": "Language Support",
    "lingua": "Language Support",
    "italiano": "Language Support",
    "trasporto": "Transport Support",
    "transport": "Transport Support",
    "transportation": "Transport Support",
    "ride": "Transport Support",
    "volunteer": "Community Support",
}

SERVICE_CATEGORY_FILTERS = [
    {"value": "Education", "label": "Tutoring and study support", "icon": "book-open"},
    {"value": "Housing Support", "label": "Student housing", "icon": "house"},
    {"value": "Digital Support", "label": "Digital and paperwork help", "icon": "monitor"},
    {"value": "Food Support", "label": "Errands and groceries", "icon": "shopping-basket"},
    {"value": "Care Support", "label": "Care and companionship", "icon": "heart-handshake"},
    {"value": "Language Support", "label": "Language support", "icon": "languages"},
    {"value": "Transport Support", "label": "Transport", "icon": "bus"},
]

SERVICE_ICONS = {item["value"]: item["icon"] for item in SERVICE_CATEGORY_FILTERS}

SEMANTIC_CATEGORY_TERMS = {
    "Care Support": "care caregiving companionship elderly care assistenza anziani compagnia",
    "Community Support": "community practical help supporto pratico aiuto comunità",
    "Digital Support": "digital support online forms spid moduli servizi digitali",
    "Education": "math tutoring homework study support ripetizioni matematica aiuto compiti",
    "Food Support": "grocery shopping groceries food support spesa alimentari pasti",
    "Healthcare": "healthcare health appointment pharmacy assistenza sanitaria farmacia",
    "Housing Support": "student housing room to rent rental affitto stanza orientamento affitti studenti ricerca stanza",
    "Language Support": "language support interpreter translation supporto linguistico interprete traduzione",
    "Transport Support": "transport ride lift appointment transport accompaniment trasporto passaggio accompagnamento",
}

ITALIAN_TRANSLATIONS = {
    "Language": "Lingua",
    "AMD AI": "IA AMD",
    "Local AI": "IA locale",
    "Request description": "Descrizione della richiesta",
    "Service category": "Categoria del servizio",
    "Not sure / anything": "Non so / qualsiasi servizio",
    "Care": "Assistenza",
    "Community": "Comunità",
    "Digital": "Digitale",
    "Education": "Istruzione",
    "Food": "Spesa",
    "Healthcare": "Salute",
    "Housing": "Casa",
    "Transport": "Trasporto",
    "optional": "facoltativo",
    "Type a municipality": "Scrivi un comune",
    "Written in": "Scritto in",
    "We understood": "Abbiamo capito",
    "Edit or remove any suggestion before matching.": "Modifica o rimuovi i suggerimenti prima di cercare.",
    "Add more details": "Aggiungi dettagli",
    "Numeric requirements": "Requisiti numerici",
    "For example: hours=2 hours": "Per esempio: hours=2 hours",
    "Requested spoken language": "Lingua parlata richiesta",
    "No stated language requirement": "Nessuna lingua richiesta",
    "Translation is OK": "La traduzione va bene",
    "Required": "Obbligatorio",
    "Nice to have": "Preferibile",
    "Suggested": "suggerito",
    "Remove": "Rimuovi",
    "Place": "Luogo",
    "Availability": "Disponibilità",
    "Could not read details yet. You can still submit your request.": "Non è stato possibile leggere i dettagli. Puoi comunque inviare la richiesta.",
    "Finding suitable matches...": "Ricerca delle proposte...",
    "Reading your request...": "Lettura della richiesta...",
    "Your details are used to find suitable help; contact details are not shared.": "I dettagli servono a trovare aiuto; i recapiti non vengono condivisi.",
    "An identical open request already exists.": "Esiste già una richiesta aperta identica.",
    "View existing request": "Vedi la richiesta esistente",
    "monday": "lunedì",
    "tuesday": "martedì",
    "wednesday": "mercoledì",
    "thursday": "giovedì",
    "friday": "venerdì",
    "saturday": "sabato",
    "sunday": "domenica",
    "morning": "mattina",
    "afternoon": "pomeriggio",
    "evening": "sera",
    "Skills needed": "Competenze richieste",
    "Request language": "Lingua della richiesta",
    "Urgency": "Urgenza",
    "normal": "normale",
    "soon": "a breve",
    "urgent": "urgente",
    "AI-assisted intake": "Descrivi con l'IA",
    "Names, phone numbers, emails, and street addresses are masked before text is sent to the AI service.": "Nomi, telefoni, email e indirizzi vengono oscurati prima dell'invio del testo al servizio IA.",
    "DOCUMENT HELP": "AIUTO CON I DOCUMENTI",
    "Photograph a letter or form": "Fotografa una lettera o un modulo",
    "The image is sent to the AI service and not stored. Image text is not automatically masked; use fake letters in demos and avoid real sensitive documents.": "L'immagine viene inviata al servizio IA e non viene conservata. Il testo nell'immagine non viene oscurato automaticamente: usa lettere inventate nelle demo ed evita documenti reali sensibili.",
    "Document image": "Immagine del documento",
    "What help would you like with it?": "Che tipo di aiuto ti serve?",
    "I consent to sending this image to the AMD AI service for analysis. The image will not be stored.": "Acconsento all'invio dell'immagine al servizio IA AMD per l'analisi. L'immagine non verrà conservata.",
    "Explain this document": "Spiega questo documento",
    "AI summary, please check the original document": "Riepilogo IA, controlla il documento originale",
    "Actions and dates": "Azioni e scadenze",
    "Use this as a help request": "Usa come richiesta di aiuto",
    "This may be an emergency. Call 112 now. CommunityConnect does not provide emergency or medical care.": "Potrebbe essere un'emergenza. Chiama subito il 112. CommunityConnect non fornisce assistenza medica o di emergenza.",
    "CommunityConnect cannot provide medical or legal advice. Describe a practical support need instead.": "CommunityConnect non fornisce consigli medici o legali. Descrivi invece un'esigenza pratica di supporto.",
    "The AI could not safely interpret this document. Check the original and contact the relevant service.": "L'IA non può interpretare il documento in modo sicuro. Controlla l'originale e contatta il servizio competente.",
    "Languages offered": "Lingue offerte",
    "Describe the practical support you need. AI suggestions are editable before matching.": "Descrivi il supporto pratico di cui hai bisogno. Puoi modificare i suggerimenti dell'IA prima dell'abbinamento.",
    "Why this match": "Perché questo abbinamento",
    "Similarity": "somiglianza",
    "Keyword fallback": "abbinamento locale per parole chiave",
    "language": "lingua",
    "Deadline": "Scadenza",
    "Original": "Originale",
    "Translate": "Traduci",
    "Translate a message": "Traduci un messaggio",
    "Message to translate": "Messaggio da tradurre",
    "Translate to": "Traduci in",
    "Translation": "Traduzione",
    "Italian": "Italiano",
    "English": "Inglese",
    "Spanish": "Spagnolo",
    "French": "Francese",
    "Urgency or safety concern": "Urgenza o sicurezza",
    "AI STATUS": "STATO IA",
    "Demo data": "Dati demo",
    "Last call": "Ultima chiamata",
    "Latency": "Latenza",
    "Tokens per second": "Token al secondo",
    "Not available": "Non disponibile",
    "No AI calls yet": "Nessuna chiamata IA",
    "generate": "generazione",
    "embed": "embedding",
    "My workspace": "La mia area",
    "Analytics": "Statistiche",
    "Privacy": "Privacy",
    "Log out": "Esci",
    "Home": "Home",
    "Log in": "Accedi",
    "Create account": "Crea account",
    "A SMALL IDEA WITH REAL COMMUNITY VALUE": "UN'IDEA SEMPLICE, UN VALORE PER LA COMUNITÀ",
    "Good help should be": "Trovare aiuto dovrebbe essere",
    "easy to find.": "semplice.",
    "CommunityConnect brings help seekers, skilled people, volunteers and local organizations into one thoughtful place.": "CommunityConnect mette in contatto chi cerca aiuto con studenti, volontari, professionisti e organizzazioni locali.",
    "I need help": "Cerco aiuto",
    "I want to help": "Voglio offrire aiuto",
    "A community assistance prototype for Genoa and Liguria.": "Un progetto di supporto alla comunità per Genova e la Liguria.",
    "Maths tutoring": "Ripetizioni di matematica",
    "Peer tutors · Genova": "Tutor tra studenti · Genova",
    "Student rental guidance": "Orientamento affitti per studenti",
    "Association service · Liguria": "Servizio di un'associazione · Liguria",
    "helper and organization profiles": "profili di volontari e organizzazioni",
    "Search local services": "Cerca servizi locali",
    "Search for tutoring, housing, SPID help...": "Cerca ripetizioni, affitti, aiuto SPID...",
    "Search services": "Cerca servizi",
    "View offer": "Vedi proposta",
    "service offers": "offerte di servizio",
    "BROWSE LOCAL HELP": "AIUTO VICINO A TE",
    "Browse by service": "Esplora per servizio",
    "Featured services": "Servizi in evidenza",
    "Explore all services": "Esplora tutti i servizi",
    "All services": "Tutti i servizi",
    "Tutoring and study support": "Ripetizioni e studio",
    "Student housing": "Casa per studenti",
    "Digital and paperwork help": "Aiuto digitale e pratiche",
    "Errands and groceries": "Commissioni e spesa",
    "Care and companionship": "Assistenza e compagnia",
    "Language support": "Supporto linguistico",
    "Transport": "Trasporto",
    "Individual/student": "Privato / studente",
    "Association / NGO": "Associazione / ONG",
    "Volunteer": "Volontario",
    "Offered by": "Offerto da",
    "Search results": "Risultati della ricerca",
    "No services match your search.": "Nessun servizio corrisponde alla ricerca.",
    "Clear filters": "Azzera filtri",
    "Request this service": "Richiedi questo servizio",
    "Open your workspace": "Apri la tua area",
    "Browse services": "Esplora i servizi",
    "Service listings from students, volunteers and local organizations.": "Servizi offerti da studenti, volontari e organizzazioni locali.",
    "Search by service, area or provider.": "Cerca per servizio, zona o organizzazione.",
    "HOW IT WORKS": "COME FUNZIONA",
    "From a need to a human connection.": "Da un bisogno a un contatto concreto.",
    "Tell us what you need": "Raccontaci di cosa hai bisogno",
    "Describe your situation in everyday words.": "Descrivi la tua situazione con parole semplici.",
    "We understand": "Individuiamo il servizio",
    "AI-assisted intake suggests a category, urgency, skills and availability.": "L'inserimento assistito dall'IA suggerisce categoria, urgenza, competenze e disponibilità.",
    "See suitable matches": "Scopri le proposte adatte",
    "Compare scores based on four factors.": "Confronta i punteggi basati su quattro fattori.",
    "Connect": "Mettiti in contatto",
    "Send a request and make contact after acceptance.": "Invia una richiesta e condividi i contatti dopo l'accettazione.",
    "BUILT FOR STUDENTS AND LOCAL SERVICES": "PER STUDENTI E SERVIZI DEL TERRITORIO",
    "Find a useful service.": "Trova un servizio utile.",
    "Offer what you know.": "Condividi ciò che sai fare.",
    "Examples include peer tutoring, student rental guidance, digital form support, errands, and services offered by local associations and NGOs in Genova and Liguria.": "Esempi: ripetizioni tra studenti, orientamento per gli affitti, supporto digitale e pratico, servizi di associazioni e ONG a Genova e in Liguria.",
    "STEP 03 / SMART MATCHING": "FASE 03 / ABBINAMENTO",
    "Best matches for your request": "Le proposte più adatte alla tua richiesta",
    "A transparent match score compares similarity, location, availability and language.": "Il punteggio confronta somiglianza, zona, disponibilità e lingua.",
    "Why this score: skill": "Dettaglio punteggio: competenze",
    "location": "zona",
    "availability": "disponibilità",
    "experience": "esperienza",
    "match": "corrispondenza",
    "Request help": "Richiedi aiuto",
    "No close matches yet": "Nessuna proposta adatta per ora",
    "Try adding a specific skill, day or service to your request.": "Prova a specificare una competenza, un giorno o un servizio.",
    "Back to workspace": "Torna alla tua area",
    "ADMIN CONSOLE / SUPABASE DATA": "AREA AMMINISTRATORE / DATI SUPABASE",
    "Community overview": "Panoramica della comunità",
    "Current profile, request and connection records.": "Profili, richieste e contatti registrati.",
    "● Live data": "● Dati aggiornati",
    "Registered users": "Utenti registrati",
    "Help seekers": "Persone che cercano aiuto",
    "Helpers": "Persone disponibili",
    "Volunteers": "Volontari",
    "NGOs": "Associazioni e ONG",
    "Successful matches": "Contatti accettati",
    "Pending requests": "Richieste in attesa",
    "Needs review": "Da verificare",
    "Most requested categories": "Servizi più richiesti",
    "Demand": "Richieste",
    "Profile review": "Revisione profili",
    "Manual review required": "Richiede una verifica manuale",
    "No profiles need review right now.": "Nessun profilo da verificare al momento.",
    "Hello,": "Ciao,",
    "Your profile is ready to make a useful connection.": "Il tuo profilo è pronto per creare un contatto utile.",
    "START A REQUEST": "INVIA UNA RICHIESTA",
    "What kind of help do you need?": "Di quale aiuto hai bisogno?",
    "Write naturally. The prototype identifies a service and day using keywords.": "Descrivi la richiesta con parole semplici. Il prototipo individua servizio e giorno tramite parole chiave.",
    "Preferred location": "Zona preferita",
    "Find suitable matches": "Trova proposte adatte",
    "YOUR PROFILE": "IL TUO PROFILO",
    "Self-reported": "Dichiarato dall'utente",
    "Experience": "Esperienza",
    "Community member": "Membro della comunità",
    "Incoming requests": "Richieste ricevute",
    "Accept": "Accetta",
    "Reject": "Rifiuta",
    "Confirm contact": "Conferma il contatto",
    "Waiting for seeker confirmation": "In attesa della conferma di chi cerca aiuto",
    "Contact details": "Dettagli di contatto",
    "No incoming requests yet. Your profile is searchable.": "Non hai ancora ricevuto richieste. Il tuo profilo è visibile nelle ricerche.",
    "Your requests": "Le tue richieste",
    "Sent requests": "Richieste inviate",
    "ACCOUNT": "ACCOUNT",
    "Delete your account and data": "Elimina account e dati",
    "This permanently removes your profile and related requests/connections.": "Questa operazione elimina definitivamente il profilo, le richieste e i contatti associati.",
    "Current password": "Password attuale",
    "I understand this cannot be undone.": "Capisco che l'operazione non può essere annullata.",
    "Delete account and data": "Elimina account e dati",
    "PRIVACY NOTICE": "INFORMATIVA SULLA PRIVACY",
    "Your data and choices": "I tuoi dati e le tue scelte",
    "What this prototype stores": "Quali dati raccoglie il prototipo",
    "Why": "Perché",
    "Contact sharing": "Condivisione dei contatti",
    "Your choices": "Le tue scelte",
    "Security and limitations": "Sicurezza e limiti",
    "MyMemory's free translation service processes interface, profile, and request text when translation is needed. Text is sent by POST. Names and contact details are not sent. Anonymous usage is limited to 5,000 characters per day.": "Il servizio di traduzione gratuito MyMemory elabora il testo dell'interfaccia, dei profili e delle richieste quando serve una traduzione. Il testo viene inviato tramite POST. Nomi e recapiti non vengono inviati. L'uso anonimo è limitato a 5.000 caratteri al giorno.",
    "Back to CommunityConnect": "Torna a CommunityConnect",
    "Connection details · CommunityConnect": "Dettagli del contatto · CommunityConnect",
    "CONNECTION ACTIVE": "CONTATTO ATTIVO",
    "Both participants have agreed to connect. Arrange details directly and use your judgment when meeting.": "Entrambe le persone hanno accettato il contatto. Concordate i dettagli direttamente e usate prudenza quando vi incontrate.",
    "No phone number provided": "Numero di telefono non disponibile",
    "REQUEST NOT COMPLETED": "RICHIESTA NON COMPLETATA",
    "We couldn’t complete that request.": "Non è stato possibile completare la richiesta.",
    "Return home": "Torna alla pagina iniziale",
    "This request could not be verified. Refresh the page and try again.": "Non è stato possibile verificare la richiesta. Ricarica la pagina e riprova.",
    "You are not allowed to access this page or action.": "Non hai l'autorizzazione per accedere a questa pagina o eseguire questa azione.",
    "The page or connection could not be found.": "La pagina o il contatto non sono stati trovati.",
    "That action is not available for this page.": "Questa azione non è disponibile in questa pagina.",
    "Something went wrong. Please try again.": "Si è verificato un errore. Riprova.",
    "Log in to continue.": "Accedi per continuare.",
    "Join the local help network.": "Entra nella rete di supporto locale.",
    "CREATE YOUR PROFILE": "CREA IL TUO PROFILO",
    "WELCOME BACK": "BENTORNATO",
    "Name": "Nome",
    "Email": "Email",
    "Password": "Password",
    "Phone (optional)": "Telefono (facoltativo)",
    "Location": "Zona",
    "Account type": "Tipo di account",
    "I can provide a skill": "Posso offrire una competenza",
    "I want to volunteer": "Voglio fare volontariato",
    "I represent an NGO": "Rappresento un'associazione o ONG",
    "Availability": "Disponibilità",
    "Skills or help offered": "Competenze o servizi offerti",
    "Short introduction": "Breve presentazione",
    "Tell the community how you can help, or what you need.": "Racconta alla comunità come puoi aiutare o di cosa hai bisogno.",
    "I consent to the storage of my profile data as described in the": "Acconsento alla conservazione dei dati del profilo come descritto nell'",
    "privacy notice": "informativa sulla privacy",
    "Already have an account?": "Hai già un account?",
    "Demo account passwords are configured by the project owner.": "La password demo è configurata dal responsabile del progetto.",
    "Create account": "Crea account",
    "Your profile is ready to make a useful connection.": "Il tuo profilo è pronto per creare un contatto utile.",
    "That email is already registered. Please log in instead.": "Questa email è già registrata. Accedi invece di creare un account.",
    "Email or password is incorrect, or email confirmation is still required.": "Email o password non corretti, oppure è ancora necessaria la conferma dell'email.",
    "Please log in to continue.": "Accedi per continuare.",
    "Education": "Istruzione",
    "Digital Support": "Supporto digitale",
    "Food Support": "Supporto per la spesa",
    "Healthcare": "Salute",
    "Care Support": "Assistenza alla persona",
    "Housing Support": "Supporto abitativo",
    "Language Support": "Supporto linguistico",
    "Transport Support": "Trasporto",
    "Community Support": "Supporto alla comunità",
    "seeker": "persona che cerca aiuto",
    "helper": "persona disponibile",
    "volunteer": "volontario",
    "ngo": "associazione o ONG",
    "WORKSPACE": "AREA PERSONALE",
    "Tell us": "Raccontaci",
    "Understand": "Comprendiamo",
    "Match": "Abbina",
    "Available": "Disponibilità:",
    "flagged": "segnalati",
    "Requests will appear here as seekers use the platform.": "Le richieste appariranno qui man mano che le persone useranno la piattaforma.",
    "I can provide a skill": "Posso offrire una competenza",
    "I represent an NGO": "Rappresento un'associazione o ONG",
    "Sunday / Weekends": "Domenica / fine settimana",
    "Maths tutoring, errands...": "Ripetizioni, commissioni...",
    "Describe a service you need, e.g. maths tutoring on Saturday.": "Descrivi il servizio che cerchi, ad esempio ripetizioni di matematica nel fine settimana.",
    "open": "aperta",
    "pending": "in attesa",
    "accepted": "accettata",
    "rejected": "rifiutata",
    "With your consent, CommunityConnect stores your name, email address, optional phone number, location, role, skills, availability, experience, and introduction. Help requests and connection statuses are also stored. Supabase Auth processes your email and password; the app does not store your password itself. Before text is sent to AMD AI, names, phone numbers, emails, and street addresses are masked. Document images are sent to AMD AI only after separate consent and are not stored; image text is not automatically masked, so use fake letters in demos.": "Con il tuo consenso, CommunityConnect conserva nome, email, numero di telefono facoltativo, zona, ruolo, competenze, lingue, disponibilità, esperienza e presentazione. Conserva anche le richieste di aiuto e lo stato dei contatti. Supabase Auth gestisce email e password: l'app non conserva la password. Prima dell'invio del testo all'IA AMD, nomi, telefoni, email e indirizzi vengono oscurati. Le immagini dei documenti sono inviate all'IA AMD solo con consenso separato e non vengono conservate; il testo nell'immagine non viene oscurato automaticamente, quindi nelle demo usa lettere inventate.",
    "Before text is sent to external AI or translation providers, common names, phone numbers, emails, and street addresses are masked. Placeholders are restored locally after a response.": "Prima dell'invio del testo ai servizi IA o di traduzione, nomi, telefoni, email e indirizzi vengono oscurati. I segnaposto vengono ripristinati localmente dopo la risposta.",
    "Uploaded document images are sent to AMD AI only after separate consent and are never saved. Image contents are not automatically de-identified, so avoid real sensitive letters and use fake documents in demos.": "Le immagini dei documenti vengono inviate all'IA AMD solo con consenso separato e non vengono mai salvate. Il contenuto delle immagini non viene anonimizzato automaticamente: evita lettere reali con dati sensibili e usa documenti inventati nelle demo.",
    "These details are used to show profiles to people seeking help, compare potential matches, and manage requests. This is a hackathon prototype, not a legal determination of GDPR compliance.": "Questi dati servono a mostrare i profili a chi cerca aiuto, confrontare possibili abbinamenti e gestire le richieste. Questo è un prototipo per un hackathon e non costituisce una valutazione legale di conformità al GDPR.",
    "Email and phone details are shown only after both participants explicitly accept the connection. Do not include highly sensitive information in a request description.": "Email e telefono sono visibili solo dopo che entrambe le persone hanno accettato esplicitamente il contatto. Non inserire informazioni molto sensibili nella descrizione.",
    "You can decline consent by not registering. While signed in, use “Delete your account and data” in your workspace to remove your Auth account, profile, requests, and associated connections. The demo database may contain synthetic profiles.": "Puoi negare il consenso semplicemente non registrandoti. Dopo l'accesso, usa “Elimina account e dati” nella tua area per rimuovere l'account, il profilo, le richieste e i contatti associati. Il database demo può contenere profili sintetici.",
    "Credentials are handled by Supabase Auth. Profile and request data are accessed by the Flask server using a server-side secret key. This prototype has no independent identity verification, data-retention schedule, or support process.": "Le credenziali sono gestite da Supabase Auth. Il server Flask accede ai profili e alle richieste con una chiave riservata lato server. Questo prototipo non offre verifica indipendente dell'identità, tempi definiti di conservazione o un servizio di assistenza.",
    "Privacy notice · CommunityConnect": "Informativa sulla privacy · CommunityConnect",
    "Home": "Home",
    "Connection details · CommunityConnect": "Dettagli del contatto · CommunityConnect",
    "Contact details": "Dettagli di contatto",
}


_ENGLISH_DEMO_CONTENT = {
    "ripetizioni di matematica": "Math tutoring",
    "ripetizioni di matematica, supporto allo studio": "Math tutoring and study support",
    "orientamento pratico per affitti studenti": "Practical guidance for student rentals",
    "orientamento affitto studenti": "Student rental guidance",
    "orientamento affitto studenti, ricerca stanza, moduli": "Student rental guidance, room search, and forms",
    "supporto digitale, spid, moduli universitari": "Digital support, SPID, and university forms",
    "spesa, commissioni, accompagnamento anziani, trasporto sociale": "Groceries, errands, elder accompaniment, and community transport",
    "ripetizioni di matematica e aiuto compiti": "Math tutoring and homework help",
    "supporto spid, email e servizi digitali": "SPID, email, and digital services support",
    "spesa e piccole commissioni": "Groceries and small errands",
    "compagnia e accompagnamento anziani": "Companionship and accompaniment for older adults",
    "traduzione e conversazione italiano-inglese": "Italian-English translation and conversation",
    "accompagnamento con i mezzi a appuntamenti": "Public transit accompaniment to appointments",
    "compilazione pratica di moduli": "Practical help completing forms",
    "aiuto per organizzare visite e appuntamenti": "Help organizing visits and appointments",
    "studente universitario offre ripetizioni e aiuto compiti.": "University student offering tutoring and homework help.",
    "sportello informativo per ricerca stanza e compilazione moduli; non offre consulenza legale.": "Information desk for finding a room and completing forms; no legal advice.",
    "studenti volontari offrono supporto pratico per servizi digitali e moduli.": "Student volunteers offering practical help with digital services and forms.",
    "associazione locale offre compagnia, spesa e accompagnamento pratico.": "Local association offering companionship, grocery shopping, and practical accompaniment.",
    "offro ripetizioni tra studenti e supporto allo studio.": "I offer peer tutoring and study support.",
    "aiuto pratico per accedere a servizi digitali e usare lo smartphone.": "Practical help accessing digital services and using smartphones.",
    "disponibile per fare la spesa e piccole commissioni di quartiere.": "Available for grocery shopping and small neighborhood errands.",
    "offro compagnia e accompagnamento per attivita quotidiane.": "I offer companionship and help with everyday activities.",
    "aiuto a organizzare documenti e cercare informazioni sugli affitti.": "I help organize documents and find information about rentals.",
    "supporto linguistico e conversazione, senza consulenza professionale.": "Language support and conversation, without professional advice.",
    "aiuto a pianificare il tragitto e accompagno con i mezzi pubblici.": "I help plan routes and accompany people on public transport.",
    "supporto pratico per leggere e compilare moduli amministrativi.": "Practical help reading and completing administrative forms.",
    "supporto logistico per organizzare appuntamenti e spostamenti.": "Logistical help organizing appointments and travel.",
    "pomeriggi": "Afternoons",
    "pomeriggi e weekend": "Afternoons and weekends",
    "weekend": "Weekends",
    "lunedi-venerdi": "Monday-Friday",
    "flessibile": "Flexible",
    "mattine": "Mornings",
}


@lru_cache(maxsize=1024)
def _translate_content_cached(text, language):
    now = monotonic()
    with _translation_backoff_lock:
        cooldown_until = _translation_backoff_until.get(language, 0)
        if cooldown_until > now:
            raise ValueError("Translation service is cooling down after a rate limit.")
        if cooldown_until:
            _translation_backoff_until.pop(language, None)
    payload = urlencode({"q": text, "langpair": f"autodetect|{language}", "mt": "1"}).encode("utf-8")
    api_request = Request(
        "https://api.mymemory.translated.net/get",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    timeout = 2
    if has_request_context():
        deadline = getattr(g, "_translation_deadline", None)
        if deadline is not None:
            timeout = min(timeout, deadline - monotonic())
            if timeout <= 0:
                raise ValueError("Translation request budget is exhausted.")
    try:
        with urlopen(api_request, timeout=timeout) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        if error.code == 429:
            with _translation_backoff_lock:
                _translation_backoff_until[language] = monotonic() + _TRANSLATION_BACKOFF_SECONDS
        raise
    if result.get("responseStatus") == 429 or result.get("quotaFinished"):
        with _translation_backoff_lock:
            _translation_backoff_until[language] = monotonic() + _TRANSLATION_BACKOFF_SECONDS
        raise ValueError("Translation service is rate-limited or out of quota.")
    if result.get("responseStatus") != 200:
        raise ValueError("Translation service did not return a translation.")
    translated = result["responseData"]["translatedText"]
    return unescape(translated)


def split_translation_text(text, max_bytes=450):
    chunks = []
    current = ""
    for word in re.findall(r"\S+\s*|\s+", text):
        if len(word.encode("utf-8")) > max_bytes:
            if current:
                chunks.append(current)
                current = ""
            piece = ""
            for character in word:
                if len((piece + character).encode("utf-8")) > max_bytes:
                    chunks.append(piece)
                    piece = ""
                piece += character
            current = piece
        elif len((current + word).encode("utf-8")) > max_bytes:
            chunks.append(current)
            current = word
        else:
            current += word
    if current:
        chunks.append(current)
    return chunks


def translate_content(text, language):
    if not text or not text.strip():
        return text
    if language == "en":
        local_translation = _ENGLISH_DEMO_CONTENT.get(text.strip().casefold())
        if local_translation:
            return local_translation
    if has_request_context() and not hasattr(g, "_translation_deadline"):
        g._translation_deadline = monotonic() + _TRANSLATION_REQUEST_BUDGET_SECONDS
    safe_text, redactions = deidentify_text(text)
    try:
        translations = [_translate_content_cached(chunk, language) for chunk in split_translation_text(safe_text)]
        return restore_redactions(" ".join(part.strip() for part in translations), redactions)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return text


def get_auth_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    if not url or not key:
        raise RuntimeError("Supabase Auth configuration is missing.")
    return create_client(url, key)


def get_service_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise RuntimeError("Supabase server configuration is missing.")
    return create_client(url, key)


def response_rows(response):
    return response.data or []


def get_one(table, **filters):
    query = get_service_client().table(table).select("*")
    for field, value in filters.items():
        query = query.eq(field, value)
    response = query.maybe_single().execute()
    return response.data if response else None


def valid_email(email):
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email))


def create_csrf_token():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_urlsafe(32)
    return session["_csrf_token"]


@app.context_processor
def inject_csrf_token():
    language = session.get("language", "it")
    if request.endpoint == "request_help":
        language_next = url_for("dashboard")
    else:
        language_next = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path

    def translate_interface(text):
        if language != "it":
            return text
        return ITALIAN_TRANSLATIONS.get(text) or translate_content(text, language)

    matching_evaluation_path = os.path.join(os.path.dirname(__file__), "data", "matching_evaluation.json")
    try:
        with open(matching_evaluation_path, encoding="utf-8") as evaluation_file:
            matching_evaluation = json.load(evaluation_file)
    except FileNotFoundError:
        matching_evaluation = None

    return {
        "csrf_token": create_csrf_token,
        "ai_status": get_backend_status(),
        "matching_evaluation": matching_evaluation,
        "language": language,
        "language_next": language_next,
        "tr": translate_interface,
        "translate_content": lambda text: translate_content(text, language),
    }


@app.route("/language/<language>")
def select_language(language):
    if language not in {"en", "it"}:
        abort(404)
    session["language"] = language
    target = request.args.get("next", "/")
    if not target.startswith("/") or target.startswith("//") or "\\" in target:
        target = url_for("index")
    response = redirect(target)
    response.set_cookie(
        LANGUAGE_PREFERENCE_COOKIE,
        language,
        max_age=60 * 60 * 24 * 365,
        secure=app.config["SESSION_COOKIE_SECURE"],
        httponly=True,
        samesite="Lax",
        path="/",
    )
    return response


@app.before_request
def protect_requests():
    if not app.secret_key or len(app.secret_key) < 32 or app.secret_key == "generate-a-unique-random-value":
        abort(503)
    if request.endpoint != "static":
        required = ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_SECRET_KEY")
        if any(not os.environ.get(key) for key in required):
            abort(503)
    preferred_language = request.cookies.get(LANGUAGE_PREFERENCE_COOKIE)
    if preferred_language in {"it", "en"}:
        session["language"] = preferred_language
    if request.method == "POST":
        expected = session.get("_csrf_token", "")
        submitted = request.form.get("csrf_token", "")
        if not expected or not hmac.compare_digest(expected, submitted):
            if request.endpoint == "login" and session.get("user_id"):
                return redirect(url_for("dashboard"))
            abort(400)


@app.errorhandler(400)
def bad_request(_error):
    return render_template("error.html", status_code=400, message="This request could not be verified. Refresh the page and try again."), 400


@app.errorhandler(403)
def forbidden(_error):
    return render_template("error.html", status_code=403, message="You are not allowed to access this page or action."), 403


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", status_code=404, message="The page or connection could not be found."), 404


@app.errorhandler(503)
def unavailable(_error):
    return render_template("error.html", status_code=503, message="Server setup is incomplete. Check the local .env settings and Supabase schema."), 503


@app.errorhandler(HTTPException)
def handle_http_error(error):
    code = error.code or 500
    messages = {
        400: "This request could not be verified. Refresh the page and try again.",
        403: "You are not allowed to access this page or action.",
        404: "The page or connection could not be found.",
        405: "That action is not available for this page.",
    }
    return render_template("error.html", status_code=code, message=messages.get(code, "Something went wrong. Please try again.")), code


@app.errorhandler(Exception)
def unexpected_error(error):
    if isinstance(error, HTTPException):
        return handle_http_error(error)
    app.logger.exception("Unhandled application error")
    return render_template("error.html", status_code=500, message="Something went wrong. Please try again."), 500


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    user = get_one("profiles", id=user_id)
    if not user:
        session.clear()
    return user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            flash("Please log in to continue.", "info")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def understand_request(text):
    ordered_words = re.findall(r"[a-z]+", text.lower())
    words = set(ordered_words)
    category = next((CATEGORIES[word] for word in ordered_words if word in CATEGORIES), "Community Support")
    skill_terms = list(dict.fromkeys(word for word in ordered_words if word in CATEGORIES))
    skill = " ".join(skill_terms[:3]) or "community assistance"
    availability = next((day for day in ["sunday", "saturday", "monday", "tuesday", "wednesday", "thursday", "friday", "weekend"] if day in text.lower()), "Flexible")
    return category, skill, availability


def _fold_request_text(text):
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(character for character in decomposed if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def _place_mentioned_in(text):
    folded = _fold_request_text(text)
    if re.search(r"\bgenoa\b", folded):
        return load_place_catalog().get("010025")
    for place in sorted(load_place_catalog().values(), key=lambda item: len(item["name"]), reverse=True):
        name = _fold_request_text(place["name"])
        if name and re.search(rf"\b{re.escape(name)}\b", folded):
            return place
    return None


def _availability_stated_in(text):
    folded = _fold_request_text(text)
    day_terms = {
        "monday": ("monday", "mondays", "lunedi"),
        "tuesday": ("tuesday", "tuesdays", "martedi"),
        "wednesday": ("wednesday", "wednesdays", "mercoledi"),
        "thursday": ("thursday", "thursdays", "giovedi"),
        "friday": ("friday", "fridays", "venerdi"),
        "saturday": ("saturday", "saturdays", "sabato"),
        "sunday": ("sunday", "sundays", "domenica"),
    }
    day_mentions = []
    for day, terms in day_terms.items():
        for term in terms:
            day_mentions.extend((match.start(), day) for match in re.finditer(rf"\b{term}\b", folded))
    for match in re.finditer(r"\b(?:weekend|weekends|fine settimana)\b", folded):
        day_mentions.extend(((match.start(), "saturday"), (match.start(), "sunday")))
    day_mentions.sort()

    part_mentions = []
    for part, terms in {
        "morning": ("morning", "mornings", "mattina", "mattine", "mattino"),
        "afternoon": ("afternoon", "afternoons", "pomeriggio", "pomeriggi"),
        "evening": ("evening", "evenings", "sera", "serata", "serate"),
    }.items():
        for term in terms:
            part_mentions.extend((match.start(), match.end(), part) for match in re.finditer(rf"\b{term}\b", folded))
    for match in re.finditer(r"\b(?:dalle?|at|from)\s+(\d{1,2})(?::\d{2})?\b", folded):
        hour = int(match.group(1))
        part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        part_mentions.append((match.start(), match.end(), part))
    part_mentions.sort()
    if not day_mentions or not part_mentions:
        return []

    slots = set()
    active_days = []
    previous_part_end = 0
    for part_start, part_end, part in part_mentions:
        days_before_part = [
            day for position, day in day_mentions
            if previous_part_end <= position < part_start
        ]
        if days_before_part:
            active_days = list(dict.fromkeys(days_before_part))
        elif not active_days:
            active_days = list(dict.fromkeys(
                day for position, day in day_mentions if position < part_start
            ))
        slots.update((day, part) for day in active_days)
        previous_part_end = part_end

    return [
        {"day": day, "part_of_day": part}
        for day in AVAILABILITY_DAYS
        for part in AVAILABILITY_PARTS
        if (day, part) in slots
    ]


def _spoken_language_stated_in(text):
    folded = _fold_request_text(text)
    patterns = (
        r"\b(?:speak|speaks|speaker|speaking|parlare|parlo|parla|parlante)\s+(?:in\s+)?(english|inglese|italian|italiano)\b",
        r"\b(english|inglese|italian|italiano)\s+(?:speaking|speaker)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, folded)
        if match:
            return "en" if match.group(1) in {"english", "inglese"} else "it"
    return None


def _numeric_values_stated_in(text):
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(character for character in decomposed if not unicodedata.combining(character))
    results = []
    patterns = (
        ("hours", r"\b(?P<low>\d+(?:[.,]\d+)?)(?:\s*(?:-|to|a)\s*(?P<high>\d+(?:[.,]\d+)?))?\s*(?:hours?|hrs?|ore)\b", "hours"),
        ("distance", r"\b(?P<low>\d+(?:[.,]\d+)?)(?:\s*(?:-|to|a)\s*(?P<high>\d+(?:[.,]\d+)?))?\s*(?:km|kilometers?|kilometres?)\b", "km"),
        ("age", r"\b(?P<low>\d+(?:[.,]\d+)?)(?:\s*(?:-|to|a)\s*(?P<high>\d+(?:[.,]\d+)?))?\s*(?:years? old|years?|anni)\b", "years"),
        ("price", r"(?:€\s*)?(?P<low>\d+(?:[.,]\d+)?)(?:\s*(?:-|to|a)\s*(?P<high>\d+(?:[.,]\d+)?))?\s*(?:€|euros?|eur)\b", "euros"),
    )
    for field, pattern, unit in patterns:
        match = re.search(pattern, folded)
        if match:
            low = float(match.group("low").replace(",", "."))
            high_text = match.group("high")
            results.append({
                "field": field,
                "value": None if high_text else low,
                "range": [low, float(high_text.replace(",", "."))] if high_text else None,
                "unit": unit,
            })
    return results


def _skill_suggestions(text, category):
    folded = _fold_request_text(text)
    suggestions = (
        (r"\b(?:math|maths|matematica|ripetizioni|tutoring)\b", "Maths tutoring"),
        (r"\b(?:rental|rentals|rent|affitto|affitti|stanza|stanze|room)\b", "Rental guidance"),
        (r"\b(?:spid|online forms?|moduli online|digital support)\b", "Digital and form support"),
        (r"\b(?:groceries|grocery|spesa)\b", "Grocery shopping"),
        (r"\b(?:lift|ride|passaggio|trasporto)\b", "Transport accompaniment"),
        (r"\b(?:elderly|anziani|caregiving|companionship)\b", "Care and companionship"),
    )
    matches = [label for pattern, label in suggestions if re.search(pattern, folded)]
    if matches:
        return list(dict.fromkeys(matches))[:8]
    if category == "Community Support" and re.search(r"\b(?:help|aiuto|supporto|assistance)\b", folded):
        return ["Community assistance"]
    return []


def detect_request_language(text):
    return detect_user_language(text)["suggested_language"]


def local_intake(text):
    folded = _fold_request_text(text)
    words = folded.split()
    category = next((CATEGORIES[word] for word in words if word in CATEGORIES), None)
    place = _place_mentioned_in(text)
    urgent_words = {"urgent", "urgently", "asap", "subito", "urgente", "urgentemente", "immediatamente", "immediately"}
    soon_words = {"soon", "presto", "appena", "quanto prima", "as soon as possible"}
    urgency = "urgent" if any(word in words for word in urgent_words) or "right away" in folded else (
        "soon" if any(phrase in folded for phrase in soon_words) else "normal"
    )
    return {
        "category": category,
        "urgency": urgency,
        "skills": _skill_suggestions(text, category),
        "availability_slots": _availability_stated_in(text),
        "number_constraints": _numeric_values_stated_in(text),
        "language": detect_request_language(text),
        "place": place["name"] if place else None,
        "stated_lang": _spoken_language_stated_in(text),
        "summary": text.strip()[:500] or "Community assistance request",
    }


def _local_triage(text):
    lowered = text.casefold()
    return {
        "emergency": bool(EMERGENCY_PATTERN.search(lowered)),
        "medical_advice": bool(MEDICAL_ADVICE_PATTERN.search(lowered)),
        "legal_advice": bool(LEGAL_ADVICE_PATTERN.search(lowered)),
        "reason": "Possible emergency" if EMERGENCY_PATTERN.search(lowered) else "",
    }


def triage_request(text, known_names=()):
    local = _local_triage(text)
    if local["emergency"] or local["medical_advice"] or local["legal_advice"]:
        return local
    prompt = (
        "Classify this community-help request for safety. Return only JSON with exactly these keys: "
        '{"emergency": boolean, "medical_advice": boolean, "legal_advice": boolean, "reason": string}. '
        "Set medical_advice or legal_advice only when the person asks for diagnosis, treatment, legal interpretation, or legal rights. "
        "Do not provide any advice. Request: " + text
    )
    response = ask_ai(prompt, max_new_tokens=128, fallback_text=json.dumps(local), known_names=known_names)
    try:
        return validate_triage(response["text"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return local


def _matching_request_text(help_request):
    description = help_request.get("description") or ""
    skill_tags = help_request.get("skill_tags") or []
    category = help_request.get("category")
    explicit_constraints = help_request.get("explicit_constraints") or {}
    if not explicit_constraints.get("category") and category == "Community Support":
        category = local_intake(description).get("category") or category
    terms = SEMANTIC_CATEGORY_TERMS.get(category, "")
    return " ".join([description, *skill_tags, terms]).strip()


def intake_request(text, known_names=()):
    local = local_intake(text)
    fallback = json.dumps(local, ensure_ascii=False)
    prompt = (
        "Extract a community assistance request. Return only valid JSON with exactly these keys: "
        '{"category": supported category|null, "urgency": "normal"|"soon"|"urgent", "skills": [string], '
        '"availability_slots": [{"day": string, "part_of_day": string}], '
        '"number_constraints": [{"field": string, "value": number|null, "range": [number, number]|null, "unit": string}], '
        '"language": "it"|"en"|null, "place": string|null, "stated_lang": "it"|"en"|null, "summary": string}. '
        "Use only supported categories: " + ", ".join(REQUEST_CATEGORIES) + ". Use null when no category or place is stated or confidently suggested. "
        "Only include a day and part-of-day slot explicitly stated by the user. Valid days are Monday through Sunday; "
        "valid parts of day are morning, afternoon, and evening. Map 'dalle 16' to afternoon. "
        "Do not infer a day from a part of day or infer a place from a region not named. Return an empty list if no complete day/part slot is explicit. "
        "Only include numeric constraints explicitly stated by the user; preserve the unit and return either an exact value or an explicit range. "
        "Only set stated_lang when the person explicitly asks for a helper who speaks a language; do not confuse it with the language the request is written in. "
        "Set urgency to soon or urgent only when urgency words such as soon, urgent, subito, urgente, or ASAP appear; otherwise use normal. "
        "Skills and categories may be suggestions, but do not present inferred place, time, numeric values, or spoken-language requirements as facts. "
        "Never provide medical or legal advice. Request: " + text
    )
    response = ask_ai(prompt, max_new_tokens=256, fallback_text=fallback, known_names=known_names)
    try:
        result = validate_intake(response["text"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        result = local
    place = _place_mentioned_in(text)
    result["place"] = place["name"] if place else None
    result["place_id"] = place["place_id"] if place else None
    result["availability_slots"] = _availability_stated_in(text)
    result["number_constraints"] = _numeric_values_stated_in(text)
    result["stated_lang"] = _spoken_language_stated_in(text)
    result["urgency"] = local["urgency"]
    language = detect_user_language(text, result.get("language"))
    result["ai_language"] = result.get("language")
    result["language"] = language["suggested_language"]
    result["language_needs_confirmation"] = language["needs_confirmation"]
    result["language_confidence"] = language["detector_confidence"]
    result["number_constraints_text"] = _format_numeric_items(result.get("number_constraints") or [])
    result["backend"] = response.get("backend", "local")
    return result


def translate_message(text, target_language, known_names=()):
    if target_language not in TRANSLATION_LANGUAGES:
        raise ValueError("Unsupported translation language")
    target_name = TRANSLATION_LANGUAGE_NAMES[target_language]
    prompt = (
        f"Translate the message into {target_name}. Return only JSON with exactly one key: "
        '{"translation": string}. Preserve placeholders in square brackets exactly. Do not answer questions or add advice. Message: ' + text
    )
    fallback = json.dumps({"translation": text}, ensure_ascii=False)
    response = ask_ai(prompt, max_new_tokens=512, fallback_text=fallback, known_names=known_names)
    try:
        return validate_translation(response["text"])["translation"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return text


def summarize_document(image_base64, language, known_names=(), user_context=""):
    language_name = "Italian" if language == "it" else "English"
    prompt = (
        f"Read this fictional demonstration letter or form and explain only its administrative contents in simple {language_name}. "
        "Do not infer legal rights, provide legal/medical advice, or invent dates. Return only JSON with exactly these keys: "
        '{"summary": string, "actions": [{"action": string, "deadline": string}], "help_request": string, '
        '"deadline": string, "emergency": boolean, "needs_professional_advice": boolean}. '
        "Use an empty string when no deadline or help request is supported by the document. If medical/legal interpretation is needed, set needs_professional_advice true. "
        + ("User's practical question: " + user_context if user_context else "")
    )
    local = {
        "summary": "Image was not analyzed in local mode. Please describe the document instead.",
        "actions": [], "help_request": "", "deadline": "", "emergency": False,
        "needs_professional_advice": False,
    }
    response = ask_ai(
        prompt,
        image_base64=image_base64,
        max_new_tokens=512,
        fallback_text=json.dumps(local, ensure_ascii=False),
        known_names=known_names,
    )
    try:
        result = validate_document_summary(response["text"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        result = local
    result["backend"] = response.get("backend", "local")
    return result


def find_matches(help_request):
    helpers = response_rows(get_service_client().table("profiles").select("*").in_("role", list(HELPER_ROLES)).execute())
    client = get_service_client()
    client.table("match_results").delete().eq("request_id", help_request["id"]).execute()
    rejected = []
    eligible = []
    for helper in helpers:
        reasons = eligibility_reasons(help_request, helper)
        if reasons:
            rejection = {"helper_id": str(helper.get("id", "")), "reasons": reasons}
            rejected.append(rejection)
            app.logger.info("Match candidate rejected: helper_id=%s reasons=%s", rejection["helper_id"], ",".join(reasons))
        else:
            eligible.append(helper)
    if not eligible:
        return {"status": "no_eligible_helpers", "matches": [], "rejections": rejected}

    request_text = _matching_request_text(help_request)
    helper_texts = [
        " ".join([*(helper.get("skill_tags") or []), helper.get("bio") or ""]).strip()
        for helper in eligible
    ]
    try:
        embedding_result = load_or_create_embeddings_with_backend(
            [request_text, *helper_texts], EMBEDDING_MODEL_ID, EMBEDDING_MODEL_VERSION,
        )
    except Exception:
        app.logger.exception("Matching embeddings failed")
        embedding_result = None
    vectors = embedding_result.get("vectors") if embedding_result else None
    embedding_backends = embedding_result.get("backends") if embedding_result else None
    if vectors is not None and (len(vectors) != len(eligible) + 1 or not vectors[0]):
        app.logger.warning("Incomplete request embeddings for %s; using manual scoring", help_request["id"])
        vectors = None
        embedding_backends = None
    if vectors is None:
        app.logger.info("AMD embeddings unavailable for %s; using manual scoring", help_request["id"])
    places = load_place_catalog()
    scored = []
    for index, helper in enumerate(eligible, start=1):
        cosine = cosine_similarity(vectors[0], vectors[index]) if vectors else None
        backend = (
            {"request": embedding_backends[0], "helper": embedding_backends[index]}
            if cosine is not None and embedding_backends else {"request": "manual", "helper": "manual"}
        )
        components = calculate_match(
            help_request,
            helper,
            cosine=cosine,
            request_place=places.get(help_request.get("place_id")),
            helper_place=places.get(helper.get("place_id")),
            backend=backend,
            model=embedding_result["model"] if cosine is not None and embedding_result else "manual-components",
        )
        if components["confidence"] is None or components["confidence"] < 40:
            continue
        scored.append({"helper": helper, **components})

    ranked = rank_matches(scored)
    if not ranked:
        return {"status": "no_suitable_matches", "matches": [], "rejections": rejected}

    match_rows = []
    for item in ranked:
        stored_components = {key: value for key, value in item.items() if key != "helper"}
        match_rows.append({
            "request_id": help_request["id"],
            "helper_id": item["helper"]["id"],
            "score": item["confidence"],
            "score_components": stored_components,
            "rank": item["rank"],
            "ranked_above_previous_by": item["ranked_above_previous_by"],
        })
    client.table("match_results").upsert(match_rows, on_conflict="request_id,helper_id").execute()
    stored_rows = response_rows(client.table("match_results").select("*").eq("request_id", help_request["id"]).execute())
    stored_by_helper = {str(row["helper_id"]): row for row in stored_rows}
    results = []
    for item in ranked:
        row = stored_by_helper.get(str(item["helper"]["id"]))
        if row:
            results.append({"helper": item["helper"], **(row.get("score_components") or {}), "confidence": row["score"]})
    return {"status": "ok" if results else "cannot_rank", "matches": results, "rejections": rejected}


def build_service_offer(profile):
    categories = profile.get("categories") or []
    category = categories[0] if categories else None
    provider_types = {
        "helper": "Individual/student",
        "volunteer": "Volunteer",
        "ngo": "Association / NGO",
    }
    return {
        "profile": profile,
        "category": category,
        "icon": SERVICE_ICONS.get(category, "handshake"),
        "provider_type": provider_types.get(profile["role"], "Community provider"),
    }


def get_service_offers():
    profiles = response_rows(
        get_service_client().table("profiles").select("*").in_("role", list(HELPER_ROLES)).eq("active", True).order("created_at", desc=True).execute()
    )
    return [build_service_offer(profile) for profile in profiles]


def get_connections_for_user(user, as_helper):
    client = get_service_client()
    field = "helper_id" if as_helper else "seeker_id"
    if as_helper:
        connection_rows = response_rows(client.table("connections").select("*").eq("helper_id", user["id"]).order("id", desc=True).execute())
    else:
        requests = response_rows(client.table("help_requests").select("*").eq("seeker_id", user["id"]).execute())
        request_ids = [row["id"] for row in requests]
        connection_rows = []
        for request_id in request_ids:
            connection_rows.extend(response_rows(client.table("connections").select("*").eq("request_id", request_id).order("id", desc=True).execute()))
    results = []
    for connection in connection_rows:
        help_request = get_one("help_requests", id=connection["request_id"])
        if not help_request:
            continue
        other_id = help_request["seeker_id"] if as_helper else connection["helper_id"]
        other = get_one("profiles", id=other_id)
        if other:
            results.append({
                **connection,
                "description": help_request["description"],
                "seeker_name": other["name"] if as_helper else user["name"],
                "helper_name": other["name"] if not as_helper else user["name"],
            })
    return results


@app.route("/")
def index():
    user = current_user()
    offers = get_service_offers()
    return render_template(
        "index.html",
        user=user,
        public_stats={"offers": len(offers)},
        featured_services=offers[:3],
        service_categories=SERVICE_CATEGORY_FILTERS,
    )


@app.route("/services")
def services():
    user = current_user()
    query = request.args.get("q", "").strip()[:100]
    category = request.args.get("category", "").strip()
    allowed_categories = {item["value"] for item in SERVICE_CATEGORY_FILTERS}
    if category not in allowed_categories:
        category = ""

    offers = get_service_offers()
    query_category = understand_request(query)[0] if query else ""
    filtered = []
    for offer in offers:
        profile = offer["profile"]
        searchable = " ".join((
            profile.get("skills") or "",
            profile.get("bio") or "",
            profile.get("location") or "",
            profile.get("name") or "",
            offer["provider_type"],
        )).casefold()
        if category and offer["category"] != category:
            continue
        if query and query.casefold() not in searchable and offer["category"] != query_category:
            continue
        filtered.append(offer)

    return render_template(
        "services.html",
        user=user,
        offers=filtered,
        query=query,
        selected_category=category,
        service_categories=SERVICE_CATEGORY_FILTERS,
    )


def _parse_skill_tags(value):
    tags = []
    seen = set()
    for raw_tag in re.split(r"[,;\n]+", value):
        tag = raw_tag.strip()
        key = tag.casefold()
        if tag and key not in seen:
            tags.append(tag)
            seen.add(key)
    return tags


def _parse_numeric_items(value):
    items = []
    seen = set()
    number_pattern = re.compile(
        r"^\s*(?P<field>[a-zA-Z][a-zA-Z0-9 _-]{0,39})\s*=\s*"
        r"(?P<low>\d+(?:[.,]\d+)?)(?:\s*(?:-|to)\s*(?P<high>\d+(?:[.,]\d+)?))?\s*"
        r"(?P<unit>[a-zA-Zà-ÿ][a-zA-Zà-ÿ /_-]{0,23})\s*$",
        re.IGNORECASE,
    )
    for raw_item in re.split(r"[;\n]+", value):
        if not raw_item.strip():
            continue
        match = number_pattern.fullmatch(raw_item)
        if not match:
            raise ValueError("Enter numeric details as field=value unit, one per line")
        field = match.group("field").strip()
        unit = match.group("unit").strip()
        low = float(match.group("low").replace(",", "."))
        high_text = match.group("high")
        if not math.isfinite(low):
            raise ValueError("Numeric values must be finite")
        if high_text is None:
            item = {"field": field, "value": low, "range": None, "unit": unit}
        else:
            high = float(high_text.replace(",", "."))
            if not math.isfinite(high) or high < low:
                raise ValueError("Numeric ranges must be finite and increasing")
            item = {"field": field, "value": None, "range": [low, high], "unit": unit}
        key = (field.casefold(), unit.casefold())
        if key in seen:
            raise ValueError("Numeric field and unit pairs must be unique")
        seen.add(key)
        items.append(item)
    return items


def _format_numeric_items(items):
    formatted = []
    for item in items:
        value = item["value"]
        number = f"{value:g}" if value is not None else f"{item['range'][0]:g}-{item['range'][1]:g}"
        formatted.append(f"{item['field']}={number} {item['unit']}")
    return "\n".join(formatted)


def _parse_availability_slots(values):
    slots = set()
    for value in values:
        day, separator, part_of_day = value.partition("|")
        if not separator or day not in AVAILABILITY_DAYS or part_of_day not in AVAILABILITY_PARTS:
            raise ValueError("Availability contains an unsupported day or time of day")
        slots.add((day, part_of_day))
    return [
        {"day": day, "part_of_day": part_of_day}
        for day in AVAILABILITY_DAYS
        for part_of_day in AVAILABILITY_PARTS
        if (day, part_of_day) in slots
    ]


def _render_registration_form(user, status=200):
    return render_template(
        "index.html",
        user=user,
        show_register=True,
        request_categories=REQUEST_CATEGORIES,
        spoken_language_options=SPOKEN_LANGUAGE_OPTIONS,
        availability_days=AVAILABILITY_DAYS,
        availability_parts=AVAILABILITY_PARTS,
        places=place_options(),
    ), status


def _precompute_matching_embedding(text, subject):
    try:
        result = load_or_create_embeddings_with_backend(
            [text], EMBEDDING_MODEL_ID, EMBEDDING_MODEL_VERSION,
        )
    except Exception:
        app.logger.exception("Could not precompute matching embedding for %s", subject)
        return None
    if not result:
        app.logger.warning("Matching embedding unavailable for %s; requests may be unrankable", subject)
        return None
    app.logger.info(
        "Matching embedding cached for %s using %s/%s (%s)",
        subject, result["model"], result["model_version"], result["backends"][0],
    )
    return result


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        role = request.form.get("role", "").strip()
        password = request.form.get("password", "")
        bio = request.form.get("bio", "").strip()
        experience = request.form.get("experience", "").strip()
        place_id = request.form.get("place_id", "").strip()
        place = load_place_catalog().get(place_id)
        skill_tags = _parse_skill_tags(request.form.get("skill_tags", ""))
        numeric_offers_text = request.form.get("numeric_offers", "")
        categories = request.form.getlist("categories")
        spoken_languages = request.form.getlist("spoken_languages")

        if role not in ALLOWED_ROLES:
            abort(403)
        if not name or not valid_email(email):
            flash("Enter a name and valid email address.", "error")
            return _render_registration_form(current_user(), 400)
        if len(password) < 12:
            flash("Choose a password with at least 12 characters.", "error")
            return _render_registration_form(current_user(), 400)
        if request.form.get("consent") != "yes":
            flash("Please read the privacy notice and provide consent to create an account.", "error")
            return _render_registration_form(current_user(), 400)
        if phone and not re.fullmatch(r"[+0-9() .-]+", phone):
            flash("Enter a phone number using digits and standard phone symbols.", "error")
            return _render_registration_form(current_user(), 400)
        if not place:
            flash("Choose a place from the Liguria catalog.", "error")
            return _render_registration_form(current_user(), 400)
        if any(category not in ALLOWED_CATEGORIES for category in categories):
            flash("Choose categories from the supported list.", "error")
            return _render_registration_form(current_user(), 400)
        if any(language not in SPOKEN_LANGUAGE_OPTIONS for language in spoken_languages):
            flash("Choose spoken languages from the supported list.", "error")
            return _render_registration_form(current_user(), 400)
        if role in HELPER_ROLES and (not skill_tags or not categories or not spoken_languages):
            flash("Providers must add skill tags, at least one category, and a spoken language.", "error")
            return _render_registration_form(current_user(), 400)
        if any(len(value) > limit for value, limit in ((name, 80), (email, 254), (phone, 40), (bio, 500), (experience, 120))):
            flash("One or more fields are too long.", "error")
            return _render_registration_form(current_user(), 400)
        try:
            availability_slots = _parse_availability_slots(request.form.getlist("availability_slots"))
            numeric_offers = _parse_numeric_items(numeric_offers_text)
            travel_text = request.form.get("max_travel_km", "").strip()
            max_travel_km = float(travel_text) if travel_text else None
            if role in HELPER_ROLES and (max_travel_km is None or not math.isfinite(max_travel_km) or max_travel_km <= 0):
                raise ValueError("Set a positive travel distance in kilometers")
        except ValueError as error:
            flash(str(error), "error")
            return _render_registration_form(current_user(), 400)

        skills = ", ".join(skill_tags)
        languages = ", ".join(SPOKEN_LANGUAGE_OPTIONS[language] for language in spoken_languages)
        availability = ", ".join(f"{slot['day']} {slot['part_of_day']}" for slot in availability_slots)
        profile_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "location": place["name"],
            "role": role,
            "skills": skills,
            "languages": languages,
            "availability": availability,
            "skill_tags": skill_tags,
            "categories": categories,
            "spoken_languages": spoken_languages,
            "numeric_offers": numeric_offers,
            "availability_slots": availability_slots,
            "place_id": place_id,
            "max_travel_km": max_travel_km,
            "active": role in HELPER_ROLES and request.form.get("active") == "yes",
            "experience": experience,
            "bio": bio,
            "self_reported": True,
            "consent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            auth_response = get_auth_client().auth.sign_up({"email": email, "password": password})
        except Exception:
            flash("We could not create that account. Check the email or try again.", "error")
            return _render_registration_form(None, 400)
        auth_user = getattr(auth_response, "user", None)
        if not auth_user:
            flash("We could not create that account. Check the details or try another email.", "error")
            return _render_registration_form(current_user(), 400)
        user_id = str(auth_user.id)
        profile = {
            "id": user_id,
            **profile_data,
        }
        try:
            get_service_client().table("profiles").insert(profile).execute()
        except Exception:
            get_auth_client().auth.admin.delete_user(user_id)
            app.logger.exception("Profile creation failed after Auth signup")
            flash("We could not finish creating your profile. Please try again.", "error")
            return _render_registration_form(None, 503)
        if role in HELPER_ROLES:
            _precompute_matching_embedding(" ".join([*skill_tags, bio]).strip(), f"profile {user_id}")
        session.clear()
        auth_session = getattr(auth_response, "session", None)
        if auth_session:
            session["user_id"] = user_id
            session.permanent = True
            return redirect(url_for("dashboard"))
        flash("Account created. Confirm your email, then log in.", "success")
        return redirect(url_for("login"))
    return _render_registration_form(current_user())[0]


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET" and current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        try:
            auth_response = get_auth_client().auth.sign_in_with_password({"email": email, "password": password})
            auth_user = getattr(auth_response, "user", None)
            if auth_user:
                profile = get_one("profiles", id=str(auth_user.id))
                if profile:
                    session.clear()
                    session["user_id"] = profile["id"]
                    session.permanent = True
                    return redirect(url_for("admin" if profile["role"] == "admin" else "dashboard"))
        except Exception:
            pass
        flash("Email or password is incorrect, or email confirmation is still required.", "error")
    return render_template("index.html", user=current_user(), show_login=True)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/profile/active", methods=["POST"])
@login_required
def update_provider_active():
    user = current_user()
    if user["role"] not in HELPER_ROLES:
        abort(403)
    active = request.form.get("active") == "yes"
    get_service_client().table("profiles").update({"active": active}).eq("id", user["id"]).execute()
    if active:
        _precompute_matching_embedding(
            " ".join([*(user.get("skill_tags") or []), user.get("bio") or ""]).strip(),
            f"profile {user['id']}",
        )
    flash("Provider visibility updated.", "success")
    return redirect(url_for("dashboard"))


def render_seeker_dashboard(
    user, *, intake=None, safety=None, document=None, draft_description="", request_form_values=None,
):
    client = get_service_client()
    mine = response_rows(client.table("help_requests").select("*").eq("seeker_id", user["id"]).order("id", desc=True).execute())
    outgoing = get_connections_for_user(user, as_helper=False)
    return render_template(
        "index.html", user=user, mine=mine, outgoing=outgoing, incoming=[], page="dashboard",
        intake=intake, safety=safety, document=document, draft_description=draft_description,
        request_categories=REQUEST_CATEGORIES, translation_languages=TRANSLATION_LANGUAGE_NAMES,
        spoken_language_options=SPOKEN_LANGUAGE_OPTIONS,
        availability_days=AVAILABILITY_DAYS,
        availability_parts=AVAILABILITY_PARTS,
        category_short_labels={
            "Care Support": "Care", "Community Support": "Community", "Digital Support": "Digital",
            "Education": "Education", "Food Support": "Food", "Healthcare": "Healthcare",
            "Housing Support": "Housing", "Language Support": "Language", "Transport Support": "Transport",
        },
        request_form_values=request_form_values or {},
        places=place_options(),
    )


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    client = get_service_client()
    if user["role"] == "seeker":
        return render_seeker_dashboard(user)
    incoming = get_connections_for_user(user, as_helper=True)
    return render_template("index.html", user=user, incoming=incoming, outgoing=[], mine=[], page="dashboard")


@app.route("/intake", methods=["POST"])
@login_required
def smart_intake():
    user = current_user()
    if user["role"] != "seeker":
        abort(403)
    description = request.form.get("description", "").strip()
    if not description or len(description) > 1000:
        flash("Enter a request up to 1,000 characters.", "error")
        return render_seeker_dashboard(user, draft_description=description), 400
    safety = triage_request(description, (user.get("name", ""),))
    if safety["emergency"] or safety["medical_advice"] or safety["legal_advice"]:
        return render_seeker_dashboard(user, safety=safety, draft_description=description)
    intake = intake_request(description, (user.get("name", ""),))
    return render_seeker_dashboard(user, intake=intake, draft_description=intake["summary"])


@app.route("/request-intake-preview", methods=["POST"])
@login_required
def request_intake_preview():
    user = current_user()
    if user["role"] != "seeker":
        abort(403)
    description = request.form.get("description", "").strip()
    if not description or len(description) > 1000:
        return jsonify({"error": "Enter a request up to 1,000 characters."}), 400
    intake = intake_request(description, (user.get("name", ""),))
    place = load_place_catalog().get(intake.get("place_id"))
    return jsonify({
        "description_hash": hashlib.sha256(description.encode("utf-8")).hexdigest(),
        "category": intake.get("category"),
        "skills": intake.get("skills") or [],
        "availability_slots": intake.get("availability_slots") or [],
        "number_constraints": intake.get("number_constraints") or [],
        "language": intake.get("language"),
        "language_needs_confirmation": intake.get("language_needs_confirmation", True),
        "place_id": place["place_id"] if place else None,
        "place_name": place["name"] if place else None,
        "place_province": place["province"] if place else None,
        "stated_lang": intake.get("stated_lang"),
        "urgency": intake.get("urgency", "normal"),
    })


def _uploaded_image_bytes(upload):
    if not upload or upload.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
        return None
    image_bytes = upload.stream.read(10 * 1024 * 1024 + 1)
    if len(image_bytes) > 10 * 1024 * 1024:
        return None
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"
    return image_bytes if is_jpeg or is_png or is_webp else None


@app.route("/document-intake", methods=["POST"])
@login_required
def document_intake():
    user = current_user()
    if user["role"] != "seeker":
        abort(403)
    if request.form.get("document_consent") != "yes":
        flash("Consent is required before the image is sent to the AI service.", "error")
        return render_seeker_dashboard(user)
    image_bytes = _uploaded_image_bytes(request.files.get("document_image"))
    if image_bytes is None:
        flash("Choose a valid PNG, JPEG, or WebP image no larger than 10 MB.", "error")
        return render_seeker_dashboard(user)
    language = session.get("language", "it")
    context = request.form.get("document_context", "").strip()[:500]
    document = summarize_document(
        base64.b64encode(image_bytes).decode("ascii"),
        language,
        (user.get("name", ""),),
        context,
    )
    if document["emergency"] or document["needs_professional_advice"]:
        document["help_request"] = ""
    elif document["help_request"]:
        safety = triage_request(document["help_request"], (user.get("name", ""),))
        if safety["emergency"] or safety["medical_advice"] or safety["legal_advice"]:
            document["emergency"] = safety["emergency"]
            document["needs_professional_advice"] = safety["medical_advice"] or safety["legal_advice"]
            document["help_request"] = ""
    if context and document["help_request"]:
        document["help_request"] = f"{document['help_request']} {context}"[:500]
    intake = None
    if document["help_request"] and not document["emergency"] and not document["needs_professional_advice"]:
        intake = intake_request(document["help_request"], (user.get("name", ""),))
        intake["place_id"] = user.get("place_id")
    return render_seeker_dashboard(user, document=document, intake=intake, draft_description=document["help_request"])


@app.route("/request-help", methods=["POST"])
@login_required
def request_help():
    user = current_user()
    if user["role"] != "seeker":
        abort(403)
    description = request.form.get("description", "").strip()
    if not description or len(description) > 1000:
        flash("Enter a request up to 1,000 characters.", "error")
        return redirect(url_for("dashboard"))
    form_values = {
        "intake_preview_hash": request.form.get("intake_preview_hash", ""),
        "description": description,
        "category": request.form.get("category", ""),
        "category_choice": request.form.get("category_choice", ""),
        "category_source": request.form.get("category_source", ""),
        "category_required": request.form.get("category_required") == "yes",
        "place_id": request.form.get("place_id", ""),
        "place_query": request.form.get("place_query", ""),
        "place_source": request.form.get("place_source", ""),
        "place_required": request.form.get("place_required") == "yes",
        "skill_tags": request.form.get("skill_tags", ""),
        "skills_required": request.form.get("skills_required") == "yes",
        "number_constraints": request.form.get("number_constraints", ""),
        "numbers_required": request.form.get("numbers_required") == "yes",
        "availability_slots": request.form.getlist("availability_slots"),
        "availability_required": request.form.get("availability_required") == "yes",
        "user_lang": request.form.get("user_lang", ""),
        "user_lang_confirmed": request.form.get("user_lang_confirmed") == "yes",
        "stated_lang": request.form.get("stated_lang", ""),
        "stated_lang_required": request.form.get("stated_lang_required") == "yes",
        "urgency": request.form.get("urgency", "normal"),
        "urgency_touched": request.form.get("urgency_touched") == "yes",
        "translation_ok": request.form.get("translation_ok") == "yes",
    }
    safety = triage_request(description, (user.get("name", ""),))
    if safety["emergency"] or safety["medical_advice"] or safety["legal_advice"]:
        return render_seeker_dashboard(
            user, safety=safety, draft_description=description, request_form_values=form_values,
        )
    places = load_place_catalog()
    preview_hash = request.form.get("intake_preview_hash", "")
    current_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
    previewed = bool(preview_hash and hmac.compare_digest(preview_hash, current_hash))
    intake = None if previewed else intake_request(description, (user.get("name", ""),))

    category_choice = request.form.get("category_choice", "").strip()
    category_source = request.form.get("category_source", "").strip()
    category = category_choice or request.form.get("category", "").strip()
    if not category and intake:
        category = intake.get("category") or "Community Support"
    if category not in ALLOWED_CATEGORIES:
        category = "Community Support"
    category_required = bool(category_choice) or category_source == "confirmed-ai" or request.form.get("category_required") == "yes"

    place_id = request.form.get("place_id", "").strip()
    place_query = request.form.get("place_query", "").strip()
    if not place_id and place_query:
        normalized_query = _fold_request_text(place_query)
        place_id = next((
            place["place_id"] for place in places.values()
            if normalized_query in {
                _fold_request_text(place["name"]),
                _fold_request_text(f"{place['name']} {place['province']}"),
            }
        ), "")
    if not place_id and not place_query and intake:
        place_id = intake.get("place_id") or ""
    if place_id not in places:
        place_id = ""
    place_required = (request.form.get("place_required") == "yes") if previewed else bool(place_id)

    skills_text = request.form.get("skill_tags", "").strip()
    if not skills_text and not previewed and intake:
        skills_text = ", ".join(intake.get("skills") or [])
    skill_tags = _parse_skill_tags(skills_text)
    skills_required = request.form.get("skills_required") == "yes"

    user_lang = request.form.get("user_lang", "").strip().lower()
    if not user_lang and intake:
        user_lang = intake.get("language") or ""
    language_status = detect_user_language(description, intake.get("ai_language") if intake else None)
    if user_lang not in ALLOWED_LANGUAGES or (
        not previewed and language_status["needs_confirmation"] and not request.form.get("user_lang")
    ):
        form_values.update({
            "intake_preview_hash": preview_hash if previewed else "",
            "place_id": place_id,
            "place_query": place_query,
            "skill_tags": skills_text,
            "user_lang": "" if language_status["needs_confirmation"] and not request.form.get("user_lang") else user_lang,
            "user_lang_confirmed": not language_status["needs_confirmation"] or bool(request.form.get("user_lang")),
            "language_needs_confirmation": language_status["needs_confirmation"],
        })
        return render_seeker_dashboard(
            user, intake=intake, draft_description=description, request_form_values=form_values,
        )

    stated_lang = request.form.get("stated_lang", "").strip().lower() or (intake.get("stated_lang") if intake else None)
    translation_ok = request.form.get("translation_ok") == "yes"
    if stated_lang and stated_lang not in SPOKEN_LANGUAGE_OPTIONS:
        flash("Choose a supported spoken-language requirement.", "error")
        return render_seeker_dashboard(user, draft_description=description, request_form_values=form_values), 400
    try:
        posted_slots = request.form.getlist("availability_slots")
        if posted_slots or previewed:
            availability_slots = _parse_availability_slots(posted_slots)
        else:
            availability_slots = intake.get("availability_slots", []) if intake else []
        numbers_text = request.form.get("number_constraints", "").strip()
        if numbers_text:
            number_constraints = _parse_numeric_items(numbers_text)
        else:
            number_constraints = intake.get("number_constraints", []) if intake and not previewed else []
    except ValueError as error:
        flash(str(error), "error")
        return render_seeker_dashboard(user, draft_description=description, request_form_values=form_values), 400
    availability_required = request.form.get("availability_required") == "yes" if previewed else bool(availability_slots)
    numbers_required = request.form.get("numbers_required") == "yes" if previewed else bool(number_constraints)
    stated_lang_required = request.form.get("stated_lang_required") == "yes" if previewed else bool(stated_lang)
    urgency = request.form.get("urgency", "normal").strip().lower()
    if urgency == "normal" and not request.form.get("urgency_touched") and intake:
        urgency = intake.get("urgency", "normal")
    if urgency not in ALLOWED_URGENCY:
        urgency = "normal"
    location = places[place_id]["name"] if place_id else ""
    skill = ", ".join(skill_tags)
    availability = ", ".join(f"{slot['day']} {slot['part_of_day']}" for slot in availability_slots) or "Flexible"
    explicit_constraints = {
        "stated_lang": bool(stated_lang) and stated_lang_required,
        "place_id": bool(place_id) and place_required,
        "availability_slots": bool(availability_slots) and availability_required,
        "skills": bool(skill_tags) and skills_required,
        "category": category_required,
        "numbers": bool(number_constraints) and numbers_required,
    }
    client = get_service_client()
    existing = response_rows(client.table("help_requests").select("id").eq(
        "seeker_id", user["id"],
    ).eq("status", "open").eq("description", description).limit(1).execute())
    if existing:
        existing_id = existing[0]["id"]
        flash(str(existing_id), "duplicate_request")
        return redirect(url_for("dashboard", _anchor=f"request-{existing_id}"))
    inserted = response_rows(client.table("help_requests").insert({
        "seeker_id": user["id"], "description": description, "category": category,
        "skill": skill, "skill_tags": skill_tags, "location": location, "place_id": place_id or None,
        "availability": availability, "availability_slots": availability_slots,
        "language": user_lang, "user_lang": user_lang, "user_lang_confirmed": True,
        "stated_lang": stated_lang, "translation_ok": translation_ok,
        "explicit_constraints": explicit_constraints, "number_constraints": number_constraints,
        "emergency": False, "medical": False, "urgency": urgency,
    }).execute())
    if not inserted:
        abort(500)
    help_request = inserted[0]
    _precompute_matching_embedding(
        _matching_request_text(help_request),
        f"request {help_request['id']}",
    )
    match_result = find_matches(help_request)
    return _render_match_results(user, help_request, match_result)


def _render_match_results(user, help_request, match_result):
    return render_template(
        "index.html",
        user=user,
        page="matches",
        help_request=help_request,
        matches=match_result["matches"],
        match_status=match_result["status"],
        match_rejections=match_result["rejections"],
    )


@app.route("/request-help/<int:request_id>/allow-translation", methods=["POST"])
@login_required
def allow_translation_for_request(request_id):
    user = current_user()
    if user["role"] != "seeker":
        abort(403)
    help_request = get_one("help_requests", id=request_id, seeker_id=user["id"], status="open")
    if not help_request or not help_request.get("stated_lang") or help_request.get("translation_ok"):
        abort(404)
    get_service_client().table("help_requests").update({"translation_ok": True}).eq(
        "id", request_id,
    ).eq("seeker_id", user["id"]).execute()
    help_request["translation_ok"] = True
    return _render_match_results(user, help_request, find_matches(help_request))


@app.route("/connect/<int:request_id>/<helper_id>", methods=["POST"])
@login_required
def connect(request_id, helper_id):
    user = current_user()
    if user["role"] != "seeker":
        abort(403)
    help_request = get_one("help_requests", id=request_id, seeker_id=user["id"], status="open")
    helper = get_one("profiles", id=helper_id)
    if not help_request or not helper or helper["role"] not in HELPER_ROLES:
        abort(404)
    match_result = find_matches(help_request)
    candidate = next((item for item in match_result["matches"] if item["helper"]["id"] == helper_id), None)
    if not candidate:
        abort(400)
    client = get_service_client()
    existing = client.table("connections").select("id").eq("request_id", request_id).eq("helper_id", helper_id).maybe_single().execute().data
    if not existing:
        client.table("connections").insert({
            "request_id": request_id,
            "helper_id": helper_id,
            "score": candidate["confidence"],
            "score_components": {key: value for key, value in candidate.items() if key not in {"helper", "confidence"}},
            "status": "pending",
            "seeker_accepted": False,
            "helper_accepted": False,
        }).execute()
        flash("Help request sent. The helper can accept or reject it.", "success")
    return redirect(url_for("dashboard"))


@app.route("/connection/<int:connection_id>/<action>", methods=["POST"])
@login_required
def update_connection(connection_id, action):
    if action not in {"accepted", "rejected"}:
        abort(400)
    user = current_user()
    client = get_service_client()
    connection = get_one("connections", id=connection_id, status="pending")
    if not connection:
        abort(404)
    if user["role"] in HELPER_ROLES and connection["helper_id"] == user["id"]:
        if connection.get("helper_accepted"):
            abort(404)
        changes = {"status": "rejected" if action == "rejected" else "pending", "helper_accepted": action == "accepted"}
        update = client.table("connections").update(changes).eq("id", connection_id).eq("helper_id", user["id"]).eq("status", "pending")
    elif user["role"] == "seeker":
        help_request = get_one("help_requests", id=connection["request_id"], seeker_id=user["id"])
        if not help_request or not connection.get("helper_accepted") or connection.get("seeker_accepted"):
            abort(404)
        changes = {"status": "accepted" if action == "accepted" else "rejected", "seeker_accepted": action == "accepted"}
        update = client.table("connections").update(changes).eq("id", connection_id).eq("status", "pending")
    else:
        abort(403)
    update.execute()
    flash("Connection updated.", "success")
    return redirect(url_for("dashboard"))


@app.route("/connection/<int:connection_id>")
@login_required
def contact(connection_id):
    user = current_user()
    connection = get_one("connections", id=connection_id, status="accepted", seeker_accepted=True, helper_accepted=True)
    if not connection:
        abort(404)
    help_request = get_one("help_requests", id=connection["request_id"])
    if not help_request:
        abort(404)
    if user["id"] not in {connection["helper_id"], help_request["seeker_id"]}:
        abort(404)
    seeker = get_one("profiles", id=help_request["seeker_id"])
    helper = get_one("profiles", id=connection["helper_id"])
    return render_template(
        "contact.html", user=user, seeker=seeker, helper=helper,
        translation_languages=TRANSLATION_LANGUAGE_NAMES, connection_id=connection_id,
    )


@app.route("/connection/<int:connection_id>/translate", methods=["POST"])
@login_required
def translate_connection_message(connection_id):
    user = current_user()
    connection = get_one("connections", id=connection_id, status="accepted", seeker_accepted=True, helper_accepted=True)
    if not connection:
        abort(404)
    help_request = get_one("help_requests", id=connection["request_id"])
    if not help_request or user["id"] not in {connection["helper_id"], help_request["seeker_id"]}:
        abort(404)
    original = request.form.get("message", "").strip()
    target_language = request.form.get("target_language", "").strip().lower()
    if not original or len(original) > 1000 or target_language not in TRANSLATION_LANGUAGES:
        abort(400)
    seeker = get_one("profiles", id=help_request["seeker_id"])
    helper = get_one("profiles", id=connection["helper_id"])
    translated = translate_message(original, target_language, (seeker.get("name", ""), helper.get("name", "")))
    return render_template(
        "contact.html", user=user, seeker=seeker, helper=helper,
        translation_languages=TRANSLATION_LANGUAGE_NAMES,
        connection_id=connection_id,
        original_message=original, translated_message=translated, target_language=target_language,
    )


@app.route("/delete-account", methods=["POST"])
@login_required
def delete_account():
    user = current_user()
    password = request.form.get("password", "")
    if request.form.get("confirm_delete") != "yes" or not password:
        flash("Confirm deletion and enter your current password to continue.", "error")
        return redirect(url_for("dashboard"))
    try:
        get_auth_client().auth.sign_in_with_password({"email": user["email"], "password": password})
        get_service_client().auth.admin.delete_user(user["id"])
    except Exception:
        flash("Account deletion could not be verified. Check your password and try again.", "error")
        return redirect(url_for("dashboard"))
    session.clear()
    flash("Your account and associated requests/connections were deleted.", "success")
    return redirect(url_for("index"))


@app.route("/privacy")
def privacy():
    return render_template("privacy.html", user=current_user())


@app.route("/admin")
@login_required
def admin():
    user = current_user()
    if user["role"] != "admin":
        abort(403)
    client = get_service_client()
    users = response_rows(client.table("profiles").select("*").order("created_at", desc=True).execute())
    requests = response_rows(client.table("help_requests").select("*").execute())
    connections = response_rows(client.table("connections").select("*").execute())
    stats = {
        "total": sum(profile["role"] != "admin" for profile in users),
        "seekers": sum(profile["role"] == "seeker" for profile in users),
        "helpers": sum(profile["role"] == "helper" for profile in users),
        "volunteers": sum(profile["role"] == "volunteer" for profile in users),
        "ngos": sum(profile["role"] == "ngo" for profile in users),
        "matches": sum(item["status"] == "accepted" for item in connections),
        "pending": sum(item["status"] == "pending" for item in connections),
        "flagged": sum(profile.get("needs_review", False) for profile in users),
    }
    category_totals = {}
    for help_request in requests:
        category = help_request.get("category") or "Unspecified"
        category_totals[category] = category_totals.get(category, 0) + 1
    categories = [{"category": name, "total": total} for name, total in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)]
    return render_template("index.html", user=user, page="admin", stats=stats, users=users, categories=categories)


@app.cli.command("init-db")
def initialize_database():
    """Verify the Supabase schema is ready for this app."""
    try:
        get_service_client().table("profiles").select("id").limit(1).execute()
    except Exception as error:
        raise click.ClickException("Supabase is unreachable or schema.sql has not been applied.") from error
    click.echo("Supabase connection and schema are ready.")


@app.cli.command("create-admin")
@click.option("--email", prompt=True)
def create_admin(email):
    """Create a verified administrator using Supabase Auth."""
    email = email.strip().lower()
    if not valid_email(email):
        raise click.ClickException("Enter a valid email address.")
    password = click.prompt("Admin password", hide_input=True, confirmation_prompt=True)
    if len(password) < 12:
        raise click.ClickException("Admin passwords must have at least 12 characters.")
    try:
        created = get_service_client().auth.admin.create_user({"email": email, "password": password, "email_confirm": True})
        user_id = str(created.user.id)
        get_service_client().table("profiles").insert({
            "id": user_id, "name": "Administrator", "email": email,
            "role": "admin", "skills": "Platform administration",
            "consent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }).execute()
    except Exception as error:
        raise click.ClickException("Could not create admin. Check that the email is unused and the schema is applied.") from error
    click.echo(f"Administrator account ready for {email}.")


def _clear_demo_data(client):
    client.table("help_requests").delete().eq("is_demo", True).execute()
    demo_profiles = response_rows(client.table("profiles").select("id").eq("is_demo", True).execute())
    for profile in demo_profiles:
        try:
            client.auth.admin.delete_user(profile["id"])
        except Exception:
            client.table("profiles").delete().eq("id", profile["id"]).execute()
    return len(demo_profiles)


@app.cli.command("reset-demo")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def reset_demo(yes):
    """Delete profiles, requests, connections, and Auth users marked as demo data."""
    if not yes and not click.confirm("Delete all demo profiles, requests, and their connections?"):
        click.echo("Demo reset cancelled.")
        return
    removed = _clear_demo_data(get_service_client())
    click.echo(f"Demo data reset. Removed {removed} demo profile(s) and their related records.")


@app.cli.command("seed-demo")
def seed_demo():
    """Create 40 synthetic Genoa/Liguria helpers and 30 synthetic help requests."""
    if len(DEMO_PASSWORD) < 12:
        raise click.ClickException("COMMUNITYCONNECT_DEMO_PASSWORD must be at least 12 characters.")

    demo_profiles = [
        {"name": "Studente UniGe", "email": "riya@demo.com", "location": "Genova", "role": "seeker", "skills": "Ripetizioni di matematica", "skill_tags": ["ripetizioni di matematica"], "categories": [], "languages": "Italian", "spoken_languages": ["it"], "availability": "Pomeriggi", "availability_slots": [], "place_id": "010025", "max_travel_km": None, "active": False, "bio": "Cerco ripetizioni di matematica a Genova."},
        {"name": "Tutor universitario", "email": "aarav@demo.com", "location": "Genova", "role": "helper", "skills": "Ripetizioni di matematica, supporto allo studio", "skill_tags": ["ripetizioni di matematica", "supporto allo studio"], "categories": ["Education"], "languages": "Italian, English", "spoken_languages": ["it", "en"], "availability": "Pomeriggi e weekend", "availability_slots": [], "place_id": "010025", "max_travel_km": 10, "active": True, "experience": "2 anni", "bio": "Studente universitario offre ripetizioni e aiuto compiti."},
        {"name": "Studentessa UniGe", "email": "neha@demo.com", "location": "Genova", "role": "seeker", "skills": "Orientamento affitto studenti", "skill_tags": ["orientamento affitto studenti"], "categories": [], "languages": "Italian, English", "spoken_languages": ["it", "en"], "availability": "Flessibile", "availability_slots": [], "place_id": "010025", "max_travel_km": None, "active": False, "bio": "Cerco aiuto pratico per trovare una stanza in affitto."},
        {"name": "Sportello Casa Studenti", "email": "sameer@demo.com", "location": "Genova", "role": "ngo", "skills": "Orientamento affitto studenti, ricerca stanza, moduli", "skill_tags": ["orientamento affitto studenti", "ricerca stanza", "moduli"], "categories": ["Housing Support"], "languages": "Italian, English", "spoken_languages": ["it", "en"], "availability": "Lunedi-venerdi", "availability_slots": [], "place_id": "010025", "max_travel_km": 20, "active": True, "experience": "Servizio associativo", "bio": "Sportello informativo per ricerca stanza e compilazione moduli; non offre consulenza legale."},
        {"name": "Rete Studenti Volontari", "email": "priya@demo.com", "location": "Genova", "role": "volunteer", "skills": "Supporto digitale, SPID, moduli universitari", "skill_tags": ["supporto digitale", "SPID", "moduli universitari"], "categories": ["Digital Support"], "languages": "Italian, English", "spoken_languages": ["it", "en"], "availability": "Weekend", "availability_slots": [], "place_id": "010025", "max_travel_km": 15, "active": True, "experience": "Volontariato", "bio": "Studenti volontari offrono supporto pratico per servizi digitali e moduli."},
        {"name": "Associazione di Quartiere", "email": "ngo@demo.com", "location": "Genova", "role": "ngo", "skills": "Spesa, commissioni, accompagnamento anziani, trasporto sociale", "skill_tags": ["spesa", "commissioni", "assistenza anziani", "trasporto sociale"], "categories": ["Food Support", "Community Support", "Care Support", "Transport Support"], "languages": "Italian", "spoken_languages": ["it"], "availability": "Lunedi-venerdi", "availability_slots": [], "place_id": "010025", "max_travel_km": 20, "active": True, "experience": "Associazione locale", "bio": "Associazione locale offre compagnia, spesa e accompagnamento pratico."},
    ]
    first_names = ["Alessia", "Marco", "Giulia", "Luca", "Sofia", "Matteo", "Elena", "Davide", "Chiara", "Andrea", "Marta", "Paolo", "Sara", "Giorgio", "Francesca", "Stefano", "Anna", "Riccardo", "Laura", "Fabio", "Irene", "Simone", "Valentina", "Pietro", "Camilla", "Enrico", "Beatrice", "Tommaso", "Noemi", "Federico", "Alice", "Michele", "Silvia", "Edoardo", "Greta", "Nicola"]
    last_names = ["Rossi", "Bianchi", "Ferrari", "Romano", "Costa", "Gallo", "Conti", "Greco", "Marino", "Lombardi", "Moretti", "Barbieri"]
    offers = [
        ("Ripetizioni di matematica e aiuto compiti", "Offro ripetizioni tra studenti e supporto allo studio."),
        ("Supporto SPID, email e servizi digitali", "Aiuto pratico per accedere a servizi digitali e usare lo smartphone."),
        ("Spesa e piccole commissioni", "Disponibile per fare la spesa e piccole commissioni di quartiere."),
        ("Compagnia e accompagnamento anziani", "Offro compagnia e accompagnamento per attivita quotidiane."),
        ("Orientamento pratico per affitti studenti", "Aiuto a organizzare documenti e cercare informazioni sugli affitti."),
        ("Traduzione e conversazione italiano-inglese", "Supporto linguistico e conversazione, senza consulenza professionale."),
        ("Accompagnamento con i mezzi a appuntamenti", "Aiuto a pianificare il tragitto e accompagno con i mezzi pubblici."),
        ("Compilazione pratica di moduli", "Supporto pratico per leggere e compilare moduli amministrativi."),
        ("Aiuto per organizzare visite e appuntamenti", "Supporto logistico per organizzare appuntamenti e spostamenti."),
    ]
    offer_categories = [
        "Education", "Digital Support", "Food Support", "Care Support", "Housing Support",
        "Language Support", "Transport Support", "Community Support", "Community Support",
    ]
    roles = ["helper", "helper", "volunteer", "ngo"]
    languages = ["Italian", "Italian, English", "Italian, French", "Italian, Spanish"]
    language_codes = [["it"], ["it", "en"], ["it", "fr"], ["it", "es"]]
    for index, first_name in enumerate(first_names):
        offer, bio = offers[index % len(offers)]
        demo_profiles.append({
            "name": f"{first_name} {last_names[index % len(last_names)]}",
            "email": f"liguria-helper-{index + 1:02d}@demo.com",
            "location": "Genova",
            "role": roles[index % len(roles)],
            "skills": offer,
            "skill_tags": [offer],
            "categories": [offer_categories[index % len(offer_categories)]],
            "languages": languages[index % len(languages)],
            "spoken_languages": language_codes[index % len(language_codes)],
            "availability": ["Pomeriggi", "Weekend", "Lunedi-venerdi", "Flessibile"][index % 4],
            "availability_slots": [],
            "place_id": "010025",
            "max_travel_km": 20,
            "active": True,
            "experience": ["Volontariato locale", "1 anno", "2 anni", "Disponibilita occasionale"][index % 4],
            "bio": bio,
        })

    client = get_service_client()
    _clear_demo_data(client)
    existing_users = client.auth.admin.list_users()
    auth_by_email = {item.email.lower(): item for item in existing_users}
    user_ids = {}
    for profile in demo_profiles:
        email = profile["email"]
        auth_user = auth_by_email.get(email)
        if not auth_user:
            auth_user = client.auth.admin.create_user({"email": email, "password": DEMO_PASSWORD, "email_confirm": True}).user
        user_id = str(auth_user.id)
        user_ids[email] = user_id
        client.table("profiles").upsert({
            **profile,
            "id": user_id,
            "phone": "",
            "self_reported": True,
            "is_demo": True,
            "consent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }).execute()

    demo_requests = [
        ("Cerco ripetizioni di matematica il sabato", "San Fruttuoso", "Pomeriggi"),
        ("Aiuto con SPID e accesso al portale", "Centro", "Flessibile"),
        ("Mi serve supporto per compilare moduli online", "Albaro", "Weekend"),
        ("Cerco aiuto per la spesa settimanale", "Marassi", "Pomeriggi"),
        ("Accompagnamento con i mezzi a un appuntamento", "Nervi", "Mattine"),
        ("Compagnia per una persona anziana nel pomeriggio", "Castelletto", "Pomeriggi"),
        ("Aiuto pratico per cercare una stanza in affitto", "Centro", "Flessibile"),
        ("Supporto per tradurre una lettera in italiano", "Sampierdarena", "Weekend"),
        ("Mi serve aiuto per usare email e smartphone", "Pegli", "Pomeriggi"),
        ("Cerco un tutor per ripassare matematica", "Albaro", "Weekend"),
        ("Aiuto per fare piccole commissioni nel quartiere", "Voltri", "Mattine"),
        ("Vorrei esercitarmi a parlare inglese", "Centro", "Pomeriggi"),
        ("Supporto pratico per raccogliere documenti di affitto", "Quarto", "Flessibile"),
        ("Cerco accompagnamento per andare in farmacia", "Marassi", "Mattine"),
        ("Aiuto per compilare un modulo amministrativo", "Carignano", "Weekend"),
        ("Cerco compagnia per una passeggiata con un anziano", "Nervi", "Pomeriggi"),
        ("Supporto per installare SPID sul telefono", "Sestri Ponente", "Weekend"),
        ("Mi serve un passaggio accompagnato in autobus", "Centro", "Mattine"),
        ("Aiuto con compiti di matematica e scienze", "San Fruttuoso", "Pomeriggi"),
        ("Cerco qualcuno per ritirare la spesa", "Pegli", "Flessibile"),
        ("Supporto per organizzare un tragitto in autobus", "Castelletto", "Weekend"),
        ("Vorrei aiuto per capire come usare un servizio online", "Albaro", "Pomeriggi"),
        ("Cerco supporto linguistico italiano inglese", "Centro", "Flessibile"),
        ("Aiuto pratico per ordinare documenti di casa", "Sampierdarena", "Mattine"),
        ("Cerco una persona che accompagni mia madre a fare la spesa", "Marassi", "Pomeriggi"),
        ("Help with maths tutoring on Saturday", "Centro", "Weekends"),
        ("I need help setting up SPID on my phone", "Albaro", "Flexible"),
        ("Could someone help me carry groceries home", "Nervi", "Afternoons"),
        ("I need an interpreter for a local appointment", "Centro", "Weekdays"),
        ("Looking for practical help to find a student room", "Sampierdarena", "Flexible"),
    ]
    seekers = [profile for profile in demo_profiles if profile["role"] == "seeker"]
    for index, (description, _demo_area, availability) in enumerate(demo_requests):
        category, skill, _inferred_availability = understand_request(description)
        language = detect_user_language(description)
        client.table("help_requests").insert({
            "seeker_id": user_ids[seekers[index % len(seekers)]["email"]],
            "description": description,
            "category": category,
            "skill": skill,
            "skill_tags": [skill],
            "place_id": "010025",
            "location": "Genova",
            "availability": availability,
            "availability_slots": [],
            "language": language["suggested_language"],
            "user_lang": language["suggested_language"],
            "user_lang_confirmed": not language["needs_confirmation"],
            "stated_lang": None,
            "translation_ok": False,
            "explicit_constraints": {"place_id": True},
            "number_constraints": [],
            "emergency": False,
            "medical": False,
            "urgency": "soon" if index in {4, 13, 17, 24} else "normal",
            "status": "open",
            "is_demo": True,
        }).execute()
    click.echo("Demo data ready: 40 helper/provider profiles and 30 help requests. All are marked is_demo.")


if __name__ == "__main__":
    if not app.secret_key or len(app.secret_key) < 32 or app.secret_key == "generate-a-unique-random-value":
        raise SystemExit("Set COMMUNITYCONNECT_SECRET_KEY in .env before starting the app.")
    app.run(debug=False)
