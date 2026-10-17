-- Run once for projects that had accepted connections before explicit seeker consent.
-- Reopen each accepted connection and preserve the helper's acceptance so the seeker
-- can make a fresh, explicit decision before contact details become visible.
update public.connections
set status = 'pending',
    seeker_accepted = false,
    helper_accepted = true
where status = 'accepted';
