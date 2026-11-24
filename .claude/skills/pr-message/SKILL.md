---
name: pr-message
description: Write a pull request title and body that a reviewer can read in under a minute. Use before every `gh pr create` and `gh pr edit --body`, in this repository or a collaborator's, and whenever asked to shorten, rewrite or review a PR description.
when_to_use: Any step that opens or edits a pull request ("open a PR", "submit the PR", "update the PR description"), including the PR steps of /wt-implement, /test-campaign and /add-dataset.
---

# pr-message

A PR body is read once, by someone deciding whether to merge. It answers three questions in
order: what changes, how it was checked, and what the reviewer has to do. Anything the diff
already shows is left to the diff.

## Shape

```markdown
<One or two sentences: what changes and why. Subject and claim in the first seven words.>

## Changes
- <behavior that changes, one line>
- <three to five bullets, by behavior, not by file>

## Tested
- <what was run, with the number that came back>
- Not tested: <what was not run>

## After merging        <!-- only when the reviewer or a system must do something -->
- <the step>
```

The whole body fits on one screen: about 150 words, 25 lines at most. A new feature with an
`After merging` section may reach 250 words. A one-commit fix needs only the opening sentence
and `Tested`.

**Title.** What changes, in the repository's commit style (`fix(scope): ...`), under 70
characters, no trailing period.

## Cut

- A file-by-file or commit-by-commit list. The diff has it.
- How the work went: what was tried, the order of steps, the debugging.
- A restatement of the title, a closing summary, thanks, praise, emoji, checkboxes.
- Background the reviewer already has. Link the issue or the note in one line instead.
- Any section with nothing in it. No "N/A".

## Keep, in one line each

- The instruction file or issue the change implements.
- What a required gate asks for: the schema-change record, the impact report, the served
  datasets that rebuild ([[torchcell.datamodels.schema-change-policy]]). These are the reason
  a body may run past one screen, and the only one.
- A caveat that changes what the reviewer should check.

## Sentences

The mechanical tier of [[writing-style-guide]] applies: no em-dashes, American spelling,
verbatim quotes unaltered. Beyond that:

- **Lead with the claim.** "Adds X so that Y", not "While working on Z it became clear that".
- **Numbers, not verdicts.** "38 s for a full build", "4 of 4 BLAST programs return the
  expected hit". Never "robust", "comprehensive", "significantly", "clean".
- **Measured or labeled.** A check that was not run is listed under "Not tested", never
  implied ([[CLAUDE.md]] Evidence Discipline).
- **Plain verbs.** The code adds, removes, reads, writes, fails. It does not ensure, leverage,
  enable or streamline.
- **One fact per bullet**, with no bold lead-in label and no nested bullets.

## A collaborator's repository

The reviewer has not seen our conversation, our skills or our machines.

- Name things as their repository names them. No worktree, queue or skill vocabulary.
- Say what changes for them: what they run, what they stop doing, what happens after merge.
- Put how-to material in a file in the PR and link it. The body is not the manual.

## Before sending

Read the body once as the reviewer. Delete every sentence they would skip, then check the
word count:

```bash
gh pr view <n> --json body -q .body | wc -w
```
