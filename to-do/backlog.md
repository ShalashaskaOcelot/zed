# Backlog

Non-phased suggestions and to-do items that are NOT yet scheduled. Move an item
into a `phase_<n>.md` when it is scheduled (and delete it from here); never
implement directly from here. Completed and scheduled work is not tracked here —
see the phase files, `CHANGELOG.md`, and git history. Roughly ordered
high → low within each group.

## Medium priority

- Notebook session persistence (user 2026-07-16): restore notebooks with the
  workspace like text buffers are. TWO gaps today: (1) SAVED notebooks that
  were open are not reopened on relaunch; (2) UNSAVED/untitled notebooks are
  lost (an unsaved text buffer from Ctrl-N IS restored, and because notebooks
  don't participate, closing the last window prompts to save the notebook
  instead of silently keeping it in the session like unsaved buffers).
  Implementation direction: `workspace::SerializableItem` for
  `NotebookEditor` — saved ones re-open by path; untitled ones serialize
  their nbformat JSON to the workspace DB the way unsaved buffers store
  their text.
- Save-as dialog for notebooks (user 2026-07-16): default the file-type
  filter to something sensible (not "all files") and make sure the `.ipynb`
  extension is applied/autofilled rather than left off.

## Low priority

- Kernel picker: show the env path under Jupyter-kernel entries the way
  Python Environment entries show theirs (user 2026-07-16) — registered
  venv kernelspecs currently give no clue which directory they point at.

