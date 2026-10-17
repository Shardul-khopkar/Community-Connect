(() => {
  const roleField = document.querySelector('.modal select[name="role"]');
  const requestedRole = new URLSearchParams(window.location.search).get('role');
  const allowedRoles = ['seeker', 'helper', 'volunteer', 'ngo'];

  if (roleField && allowedRoles.includes(requestedRole)) {
    roleField.value = requestedRole;
  }

  const skillsField = document.querySelector('.modal input[name="skills"]');
  const skillsLabel = skillsField?.parentElement;
  if (roleField && skillsField && skillsLabel) {
    const language = document.documentElement.lang;
    const providerLabel = skillsLabel.firstChild?.textContent.trim() || '';
    const providerPlaceholder = skillsField.placeholder;
    const seekerLabel = language === 'it' ? 'Aiuto che cerchi (facoltativo)' : 'Help you are looking for (optional)';
    const seekerPlaceholder = language === 'it' ? 'Es. ripetizioni o aiuto per SPID' : 'For example, tutoring or help with forms';

    const updateSkillsField = () => {
      const isSeeker = roleField.value === 'seeker';
      skillsField.required = !isSeeker;
      if (skillsLabel.firstChild?.nodeType === Node.TEXT_NODE) {
        skillsLabel.firstChild.textContent = `${isSeeker ? seekerLabel : providerLabel} `;
      }
      skillsField.placeholder = isSeeker ? seekerPlaceholder : providerPlaceholder;
    };

    roleField.addEventListener('change', updateSkillsField);
    updateSkillsField();
  }

  const profileSkillsHeading = document.querySelector('.profile-panel h2');
  if (profileSkillsHeading && !profileSkillsHeading.textContent.trim()) {
    profileSkillsHeading.textContent = document.documentElement.lang === 'it'
      ? 'Nessuna preferenza di servizio'
      : 'No service preferences added';
  }

  const offerBadge = document.querySelector('.hero-badge');
  if (offerBadge && offerBadge.textContent.trim().startsWith('—')) {
    offerBadge.hidden = true;
  }

  if (document.querySelector('.modal')) {
    document.querySelector('.service-discovery')?.setAttribute('hidden', '');
  }

  if (window.lucide?.createIcons) {
    window.lucide.createIcons({ attrs: { 'stroke-width': 1.8 } });
  }
})();
