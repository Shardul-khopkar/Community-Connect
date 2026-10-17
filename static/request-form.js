(() => {
  const form = document.getElementById('request-help-form');
  if (!form) return;

  const fields = {
    description: form.querySelector('[name="description"]'),
    placeQuery: form.querySelector('[name="place_query"]'),
    placeId: form.querySelector('[name="place_id"]'),
    placeSource: form.querySelector('[name="place_source"]'),
    category: form.querySelector('[name="category"]'),
    categorySource: form.querySelector('[name="category_source"]'),
    categoryRequired: form.querySelector('[name="category_required"]'),
    skills: form.querySelector('[name="skill_tags"]'),
    numbers: form.querySelector('[name="number_constraints"]'),
    statedLanguage: form.querySelector('[name="stated_lang"]'),
    urgency: form.querySelector('[name="urgency"]'),
    previewHash: form.querySelector('[name="intake_preview_hash"]'),
    languageConfirmed: form.querySelector('[name="user_lang_confirmed"]'),
  };
  const section = document.getElementById('request-understood');
  const chipsHost = document.getElementById('request-understood-chips');
  const confirmation = form.querySelector('.language-confirmation');
  const submitButton = form.querySelector('.request-submit');
  const status = document.getElementById('request-preview-status');
  const categoryRadios = [...form.querySelectorAll('[name="category_choice"]')];
  const languageRadios = [...form.querySelectorAll('[name="user_lang"]')];
  const slotInputs = [...form.querySelectorAll('[name="availability_slots"]')];
  const placeOptions = [...form.querySelectorAll('#liguria-place-options option')];
  const chips = [];
  let previewTimer = null;
  let previewController = null;
  let programmaticCategoryChange = false;

  const strings = {
    required: form.dataset.requiredLabel,
    nice: form.dataset.niceLabel,
    suggested: form.dataset.suggestedLabel,
    remove: form.dataset.removeLabel,
  };
  const manual = {
    category: fields.categorySource.value === 'manual',
    place: fields.placeSource.value === 'manual',
    language: fields.languageConfirmed.value === 'yes',
    skills: Boolean(fields.skills.value.trim()),
    numbers: Boolean(fields.numbers.value.trim()),
    statedLanguage: Boolean(fields.statedLanguage.value),
    slots: slotInputs.some((input) => input.checked),
    urgency: form.querySelector('[name="urgency_touched"]').value === 'yes',
  };

  const fold = (value) => value.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const categoryLabel = (value) => {
    const radio = categoryRadios.find((item) => item.value === value);
    return radio?.parentElement.querySelector('span')?.textContent.trim() || value;
  };
  const dayLabels = Object.fromEntries(slotInputs.map((input) => [input.value.split('|')[0], input.dataset.dayLabel]));
  const partLabels = Object.fromEntries(slotInputs.map((input) => [input.value.split('|')[1], input.dataset.partLabel]));
  const dayAliases = {
    monday: ['monday', 'lunedi'], tuesday: ['tuesday', 'martedi'], wednesday: ['wednesday', 'mercoledi'],
    thursday: ['thursday', 'giovedi'], friday: ['friday', 'venerdi'], saturday: ['saturday', 'sabato'],
    sunday: ['sunday', 'domenica'],
  };
  const partAliases = {
    morning: ['morning', 'mattina', 'mattine', 'mattino'],
    afternoon: ['afternoon', 'pomeriggio', 'pomeriggi'],
    evening: ['evening', 'sera', 'serata', 'serate'],
  };
  const setField = (name, value) => {
    const field = form.querySelector(`[name="${name}"]`);
    if (field) field.value = value;
  };
  const setRequirement = (name, required) => setField(name, required ? 'yes' : 'no');

  function placeForText(value) {
    const normalized = fold(value);
    if (!normalized) return null;
    return placeOptions.find((option) => {
      const optionName = fold(option.value);
      const bareName = fold(option.value.split(' · ')[0]);
      return normalized === optionName || normalized === bareName;
    }) || null;
  }

  function parseSlotText(value) {
    const normalized = fold(value);
    const day = Object.keys(dayAliases).find((key) => dayAliases[key].some((alias) => new RegExp(`\\b${alias}\\b`).test(normalized)));
    const part = Object.keys(partAliases).find((key) => partAliases[key].some((alias) => new RegExp(`\\b${alias}\\b`).test(normalized)));
    return day && part ? { day, part_of_day: part } : null;
  }

  function slotText(slot) {
    return `${dayLabels[slot.day] || slot.day} ${partLabels[slot.part_of_day] || slot.part_of_day}`;
  }

  function clearKind(kind) {
    for (let index = chips.length - 1; index >= 0; index -= 1) {
      if (chips[index].kind === kind) chips.splice(index, 1);
    }
  }

  function numberText(number) {
    const amount = number.value !== null && number.value !== undefined
      ? number.value
      : `${number.range[0]}-${number.range[1]}`;
    return `${number.field}=${amount} ${number.unit}`;
  }

  function syncFieldsFromChips() {
    const ofKind = (kind) => chips.filter((chip) => chip.kind === kind);
    const categoryChip = ofKind('category')[0];
    if (categoryChip) {
      fields.category.value = categoryChip.value;
      fields.categorySource.value = categoryChip.source === 'manual' ? 'manual' : categoryChip.required ? 'confirmed-ai' : 'ai';
      fields.categoryRequired.value = categoryChip.required ? 'yes' : 'no';
      if (categoryChip.required) {
        programmaticCategoryChange = true;
        const radio = categoryRadios.find((item) => item.value === categoryChip.value);
        if (radio) radio.checked = true;
        programmaticCategoryChange = false;
      }
    } else if (!manual.category && fields.categorySource.value !== 'manual') {
      fields.category.value = '';
      fields.categorySource.value = '';
      fields.categoryRequired.value = 'no';
    }

    const placeChip = ofKind('place')[0];
    if (placeChip) {
      fields.placeQuery.value = placeChip.value;
      fields.placeId.value = placeChip.placeId || '';
      fields.placeSource.value = placeChip.source === 'manual' ? 'manual' : 'ai';
      setRequirement('place_required', placeChip.required);
    } else if (!manual.place) {
      fields.placeQuery.value = '';
      fields.placeId.value = '';
      fields.placeSource.value = '';
      setRequirement('place_required', false);
    }

    const skillChips = ofKind('skill');
    if (skillChips.length || !manual.skills) {
      fields.skills.value = skillChips.map((chip) => chip.value).join(', ');
      setRequirement('skills_required', skillChips.some((chip) => chip.required));
    }
    const numberChips = ofKind('number');
    if (numberChips.length || !manual.numbers) {
      fields.numbers.value = numberChips.map((chip) => chip.text || numberText(chip.value)).join('\n');
      setRequirement('numbers_required', numberChips.some((chip) => chip.required));
    }
    const slotChips = ofKind('slot');
    if (slotChips.length || !manual.slots) {
      const selected = new Set(slotChips.map((chip) => `${chip.value.day}|${chip.value.part_of_day}`));
      slotInputs.forEach((input) => { input.checked = selected.has(input.value); });
      setRequirement('availability_required', slotChips.some((chip) => chip.required));
    }
    const spokenChip = ofKind('spoken')[0];
    if (spokenChip || !manual.statedLanguage) {
      fields.statedLanguage.value = spokenChip?.value || '';
      setRequirement('stated_lang_required', Boolean(spokenChip?.required));
    }
    const urgencyChip = ofKind('urgency')[0];
    if (urgencyChip && !manual.urgency) fields.urgency.value = urgencyChip.value;
  }

  function createSelect(options, selected) {
    const select = document.createElement('select');
    options.forEach(([value, label]) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.append(option);
    });
    select.value = selected;
    return select;
  }

  function renderChips() {
    chipsHost.replaceChildren();
    section.hidden = chips.length === 0;
    chips.forEach((chip) => {
      const row = document.createElement('div');
      row.className = 'request-understood-chip';
      row.dataset.kind = chip.kind;
      const kindLabel = document.createElement('span');
      kindLabel.className = 'request-chip-kind';
      kindLabel.textContent = form.dataset[`${chip.kind}Label`] || chip.kind;
      row.append(kindLabel);

      if (chip.kind === 'category') {
        const choices = categoryRadios.filter((radio) => radio.value).map((radio) => [radio.value, categoryLabel(radio.value)]);
        const editor = createSelect(choices, chip.value);
        editor.setAttribute('aria-label', categoryLabel(chip.value));
        editor.addEventListener('change', () => {
          chip.value = editor.value;
          chip.source = 'manual';
          manual.category = true;
          syncFieldsFromChips();
        });
        row.append(editor);
      } else if (chip.kind === 'spoken') {
        const choices = [...form.querySelectorAll('#request-spoken-language option')]
          .filter((option) => option.value).map((option) => [option.value, option.textContent]);
        const editor = createSelect(choices, chip.value);
        editor.setAttribute('aria-label', form.dataset.spokenLabel);
        editor.addEventListener('change', () => {
          chip.value = editor.value;
          chip.source = 'manual';
          manual.statedLanguage = true;
          syncFieldsFromChips();
        });
        row.append(editor);
      } else if (chip.kind === 'urgency') {
        const choices = [...fields.urgency.options].map((option) => [option.value, option.textContent]);
        const editor = createSelect(choices, chip.value);
        editor.setAttribute('aria-label', form.dataset.urgencyLabel);
        editor.addEventListener('change', () => {
          chip.value = editor.value;
          chip.source = 'manual';
          manual.urgency = true;
          form.querySelector('[name="urgency_touched"]').value = 'yes';
          syncFieldsFromChips();
        });
        row.append(editor);
      } else if (chip.kind === 'slot') {
        const day = createSelect(Object.keys(dayLabels).map((key) => [key, dayLabels[key]]), chip.value.day);
        const part = createSelect(Object.keys(partLabels).map((key) => [key, partLabels[key]]), chip.value.part_of_day);
        day.setAttribute('aria-label', form.dataset.slotLabel);
        part.setAttribute('aria-label', form.dataset.slotLabel);
        const updateSlot = () => {
          chip.value = { day: day.value, part_of_day: part.value };
          chip.source = 'manual';
          manual.slots = true;
          syncFieldsFromChips();
        };
        day.addEventListener('change', updateSlot);
        part.addEventListener('change', updateSlot);
        row.append(day, part);
      } else {
        const editor = document.createElement('input');
        editor.type = 'text';
        editor.value = chip.kind === 'place' ? chip.value : chip.kind === 'number' ? chip.text || numberText(chip.value) : chip.value;
        editor.setAttribute('aria-label', form.dataset[`${chip.kind}Label`] || chip.kind);
        if (chip.kind === 'place') editor.setAttribute('list', 'liguria-place-options');
        editor.addEventListener('input', () => {
          if (chip.kind === 'place') {
            manual.place = true;
            chip.source = 'manual';
            chip.value = editor.value;
            const option = placeForText(editor.value);
            chip.placeId = option?.dataset.placeId || '';
          } else if (chip.kind === 'slot') {
            manual.slots = true;
            chip.source = 'manual';
            chip.value = parseSlotText(editor.value) || chip.value;
          } else if (chip.kind === 'number') {
            manual.numbers = true;
            chip.source = 'manual';
            chip.text = editor.value;
          } else {
            if (chip.kind === 'skill') manual.skills = true;
            if (chip.kind === 'category') manual.category = true;
            chip.source = 'manual';
            chip.value = editor.value;
          }
          syncFieldsFromChips();
        });
        row.append(editor);
      }

      if (chip.source === 'suggested') {
        const suggested = document.createElement('span');
        suggested.className = 'request-chip-suggested';
        suggested.textContent = strings.suggested;
        row.append(suggested);
      }
      const required = document.createElement('button');
      required.type = 'button';
      required.className = 'request-chip-requirement';
      required.textContent = chip.required ? strings.required : strings.nice;
      required.setAttribute('aria-pressed', chip.required ? 'true' : 'false');
      required.addEventListener('click', () => {
        chip.required = !chip.required;
        if (chip.kind === 'category' && chip.required) {
          programmaticCategoryChange = true;
          const radio = categoryRadios.find((item) => item.value === chip.value);
          if (radio) radio.checked = true;
          programmaticCategoryChange = false;
        }
        if (chip.kind === 'category' && !chip.required) {
          programmaticCategoryChange = true;
          const anything = categoryRadios.find((item) => !item.value);
          if (anything) anything.checked = true;
          programmaticCategoryChange = false;
        }
        syncFieldsFromChips();
        renderChips();
      });
      row.append(required);

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'request-chip-remove';
      remove.textContent = '×';
      remove.setAttribute('aria-label', strings.remove);
      remove.addEventListener('click', () => {
        const kind = chip.kind;
        chips.splice(chips.indexOf(chip), 1);
        const manualKey = kind === 'spoken' ? 'statedLanguage' : kind === 'slot' ? 'slots' : kind === 'number' ? 'numbers' : kind === 'skill' ? 'skills' : kind;
        manual[manualKey] = false;
        if (kind === 'category') {
          fields.category.value = '';
          fields.categorySource.value = '';
          fields.categoryRequired.value = 'no';
          programmaticCategoryChange = true;
          const anything = categoryRadios.find((item) => !item.value);
          if (anything) anything.checked = true;
          programmaticCategoryChange = false;
        }
        if (kind === 'urgency') {
          fields.urgency.value = 'normal';
          form.querySelector('[name="urgency_touched"]').value = 'no';
        }
        syncFieldsFromChips();
        renderChips();
      });
      row.append(remove);
      chipsHost.append(row);
    });
  }

  function setChip(kind, value, { required = false, source = 'suggested', placeId = '' } = {}) {
    const chip = chips.find((item) => item.kind === kind && item.source === source && (
      kind === 'slot' ? item.value.day === value.day && item.value.part_of_day === value.part_of_day
        : kind === 'number' ? numberText(item.value) === numberText(value)
          : item.value === value
    ));
    if (chip) {
      chip.required = required;
      return chip;
    }
    const created = { kind, value, required, source, placeId };
    chips.push(created);
    return created;
  }

  function replaceKind(kind, values, source, required) {
    for (let index = chips.length - 1; index >= 0; index -= 1) {
      if (chips[index].kind === kind && chips[index].source === source) chips.splice(index, 1);
    }
    values.forEach((value) => setChip(kind, value, { source, required }));
    renderChips();
    syncFieldsFromChips();
  }

  function updatePlaceFromInput(source) {
    const option = placeForText(fields.placeQuery.value);
    fields.placeId.value = option?.dataset.placeId || '';
    fields.placeSource.value = source;
    if (fields.placeQuery.value.trim()) {
      clearKind('place');
      setChip('place', fields.placeQuery.value, {
        source,
        required: fields.placeQuery.dataset.required !== 'no',
        placeId: fields.placeId.value,
      });
      setRequirement('place_required', fields.placeQuery.dataset.required !== 'no');
    } else {
      clearKind('place');
      setRequirement('place_required', false);
    }
    renderChips();
  }

  function clearAutomaticSuggestions() {
    for (let index = chips.length - 1; index >= 0; index -= 1) {
      if (chips[index].source !== 'manual') chips.splice(index, 1);
    }
    if (fields.categorySource.value !== 'manual' && !manual.category) {
      fields.category.value = '';
      fields.categorySource.value = '';
      fields.categoryRequired.value = 'no';
    }
    if (fields.placeSource.value === 'ai' && !manual.place) {
      fields.placeQuery.value = '';
      fields.placeId.value = '';
      fields.placeSource.value = '';
      setRequirement('place_required', false);
    }
    if (!manual.skills) fields.skills.value = '';
    if (!manual.numbers) fields.numbers.value = '';
    if (!manual.slots) slotInputs.forEach((input) => { input.checked = false; });
    if (!manual.statedLanguage) fields.statedLanguage.value = '';
    if (!manual.language) {
      languageRadios.forEach((radio) => { radio.checked = false; });
      fields.languageConfirmed.value = 'no';
      confirmation.hidden = true;
    }
    if (!manual.urgency) fields.urgency.value = 'normal';
    if (!manual.urgency) form.querySelector('[name="urgency_touched"]').value = 'no';
    if (status) status.hidden = true;
    fields.previewHash.value = '';
    renderChips();
  }

  function applyPreview(data) {
    for (let index = chips.length - 1; index >= 0; index -= 1) {
      if (chips[index].source !== 'manual') chips.splice(index, 1);
    }
    if (!manual.category && data.category) {
      setChip('category', data.category, { source: 'suggested', required: false });
    }
    if (!manual.place && data.place_id) {
      const option = placeOptions.find((item) => item.dataset.placeId === data.place_id);
      const display = data.place_name ? `${data.place_name} · ${data.place_province}` : option?.value || '';
      setChip('place', display, { source: 'stated', required: true, placeId: data.place_id });
    }
    if (!manual.skills) (data.skills || []).forEach((skill) => setChip('skill', skill, { source: 'suggested', required: false }));
    if (!manual.slots) (data.availability_slots || []).forEach((slot) => setChip('slot', slot, { source: 'stated', required: true }));
    if (!manual.numbers) (data.number_constraints || []).forEach((number) => setChip('number', number, { source: 'stated', required: true }));
    if (!manual.statedLanguage && data.stated_lang) setChip('spoken', data.stated_lang, { source: 'stated', required: true });
    if (!manual.urgency && data.urgency && data.urgency !== 'normal') setChip('urgency', data.urgency, { source: 'stated', required: false });

    if (!manual.language) {
      languageRadios.forEach((radio) => { radio.checked = !data.language_needs_confirmation && radio.value === data.language; });
      fields.languageConfirmed.value = data.language_needs_confirmation ? 'no' : 'yes';
      confirmation.hidden = !data.language_needs_confirmation;
    }
    if (!manual.place) {
      const placeChip = chips.find((chip) => chip.kind === 'place');
      fields.placeQuery.value = placeChip?.value || '';
      fields.placeId.value = placeChip?.placeId || '';
      fields.placeSource.value = placeChip ? 'ai' : '';
    }
    if (!manual.skills) fields.skills.dataset.source = 'ai';
    if (!manual.numbers) fields.numbers.dataset.source = 'ai';
    if (!manual.urgency && data.urgency) fields.urgency.value = data.urgency;
    fields.previewHash.value = data.description_hash;
    syncFieldsFromChips();
    renderChips();
  }

  async function requestPreview() {
    const description = fields.description.value.trim();
    if (description.length < 8) return;
    if (previewController) previewController.abort();
    previewController = new AbortController();
    const data = new URLSearchParams({
      csrf_token: form.querySelector('[name="csrf_token"]').value,
      description,
    });
    try {
      if (status) {
        section.hidden = false;
        status.hidden = false;
        status.textContent = form.dataset.previewingLabel;
      }
      const response = await fetch(form.dataset.previewUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json' },
        body: data.toString(),
        signal: previewController.signal,
      });
      if (!response.ok) throw new Error('preview-failed');
      const result = await response.json();
      if (description !== fields.description.value.trim()) return;
      applyPreview(result);
      if (status) status.hidden = true;
    } catch (error) {
      if (error.name !== 'AbortError' && status) {
        section.hidden = false;
        status.hidden = false;
        status.textContent = form.dataset.previewError;
      }
    }
  }

  categoryRadios.forEach((radio) => radio.addEventListener('change', () => {
    if (programmaticCategoryChange) return;
    manual.category = true;
    clearKind('category');
    fields.category.value = radio.value;
    fields.categorySource.value = radio.value ? 'manual' : '';
    fields.categoryRequired.value = radio.value ? 'yes' : 'no';
    renderChips();
  }));

  fields.placeQuery.addEventListener('input', () => {
    manual.place = true;
    fields.placeQuery.dataset.required = fields.placeQuery.value ? 'yes' : 'no';
    updatePlaceFromInput('manual');
  });
  languageRadios.forEach((radio) => radio.addEventListener('change', () => {
    manual.language = true;
    fields.languageConfirmed.value = 'yes';
    confirmation.hidden = true;
  }));
  fields.urgency.addEventListener('change', () => {
    manual.urgency = true;
    form.querySelector('[name="urgency_touched"]').value = 'yes';
    replaceKind('urgency', fields.urgency.value === 'normal' ? [] : [fields.urgency.value], 'manual', false);
  });
  fields.skills.addEventListener('change', () => {
    manual.skills = true;
    const values = fields.skills.value.split(/[,;\n]+/).map((value) => value.trim()).filter(Boolean);
    replaceKind('skill', values, 'manual', false);
  });
  fields.numbers.addEventListener('change', () => {
    manual.numbers = true;
    const values = fields.numbers.value.split(/[;\n]+/).map((value) => value.trim()).filter(Boolean)
      .map((text) => {
        const match = text.match(/^([^=]+)=\s*(\d+(?:[.,]\d+)?)(?:\s*-\s*(\d+(?:[.,]\d+)?))?\s+(.+)$/);
        if (!match) return null;
        return {
          field: match[1].trim(), value: match[3] ? null : Number(match[2].replace(',', '.')),
          range: match[3] ? [Number(match[2].replace(',', '.')), Number(match[3].replace(',', '.'))] : null,
          unit: match[4].trim(),
        };
      }).filter(Boolean);
    replaceKind('number', values, 'manual', true);
  });
  fields.statedLanguage.addEventListener('change', () => {
    manual.statedLanguage = true;
    replaceKind('spoken', fields.statedLanguage.value ? [fields.statedLanguage.value] : [], 'manual', Boolean(fields.statedLanguage.value));
  });
  slotInputs.forEach((input) => input.addEventListener('change', () => {
    manual.slots = true;
    const values = slotInputs.filter((item) => item.checked).map((item) => {
      const [day, part_of_day] = item.value.split('|');
      return { day, part_of_day };
    });
    replaceKind('slot', values, 'manual', true);
  }));

  fields.description.addEventListener('input', () => {
    clearAutomaticSuggestions();
    clearTimeout(previewTimer);
    if (previewController) previewController.abort();
    previewTimer = setTimeout(() => requestPreview(), 650);
  });

  form.addEventListener('submit', () => {
    if (submitButton.disabled) return;
    submitButton.disabled = true;
    submitButton.setAttribute('aria-busy', 'true');
    submitButton.dataset.originalText = submitButton.textContent;
    submitButton.firstChild.textContent = form.dataset.findingLabel;
  });

  if (fields.description.value.trim()) requestPreview();
})();