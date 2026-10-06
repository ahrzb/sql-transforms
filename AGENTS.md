# Instructions for agents

These rules apply to every agent that works in this repository: the loop
sessions, their workers, and a single session. The loops have their own
rules in [`loops/README.md`](loops/README.md).

1. **Text for the owner.** Load the `simple-english` skill
   (`.claude/skills/simple-english/SKILL.md`) before you write such text.
   This is a doc that the owner reviews or maintains, a report, a decision
   record, or the summary of a large technical PR. Other text, such as code
   comments, `PLANS.md` and an ordinary PR body, needs only to be clear.
2. **Terms.** [`GLOSSARY.md`](GLOSSARY.md) holds the project's terms. Use
   them when you name a concept of the project.
3. **Questions for the owner.** Ask the owner only through a decision
   record. Before you ask, read the loop's `decisions/closed/` and
   `decisions/postponed/`. If the question is new, write it in
   `decisions/open/` and list it in the next report. Do not name an owner
   question that has no record.
