---
name: simple-english
description: Writing rules (Simplified Technical English) and the glossary rules, for text that the owner reads. Use it when you write or change a doc that the owner reviews or maintains, a report, a decision record, or the summary of a large technical PR. Not needed for code, code comments, PLANS.md, the ticket board or an ordinary PR body.
---

# Simple English

Use this skill for text that the owner reads. This is the docs that the
owner reviews or maintains, reports, decision records, and the summary of a
large technical PR. Working files for agents, such as `PLANS.md` and the ticket
board, need only be clear to the next agent.

## The writing rules

The rules come from ASD-STE100 Issue 9, Simplified Technical English (STE).
We do not use the full standard. We do not check words against its
dictionary. We use its sentence rules fully and its word rules as guidance.

### Words

1. **Use one word for one thing.** Use the term in
   [`GLOSSARY.md`](../../../GLOSSARY.md). Do not change to a synonym for variety. Do
   not use a word under _Avoid_ for the thing that its entry names.
2. **Do not make new names.** Do not write a metaphor or a private name for
   a concept (for example "yardstick", "north star", "wave 1 reading"). If
   you need a new term, add it to the glossary in the same PR and then use
   it.
3. **Use common words.** Write "use", not "utilize". Write "start", not
   "kick off". Write "about", not "circa".
4. **Use verbs for actions.** Write "measure the latency", not "perform a
   measurement of the latency".
5. **Use a single verb, not a phrasal verb,** when one exists. Write
   "remove", not "take out". Write "examine", not "look into".
6. **Use a maximum of three nouns together.** Write "the cost of a build of
   a deep CASE tree", not "deep CASE tree build cost".
7. **Do not use words that only praise or stress.** Do not write "robust",
   "seamless", "simply", "just", "clearly" or "very". Give a measurement
   instead.
8. **Explain each code name the first time.** A reader can know the term
   but not the file. Write "the pass that finds repeated subexpressions
   (`share.rs`)", not "`share.rs`" alone.
9. **Write abbreviations in full** the first time, unless the glossary has
   them.

### Sentences

1. **Keep sentences short.** An instruction has a maximum of 20 words. A
   description has a maximum of 25 words.
2. **Write one instruction or one idea in each sentence.** Do not join two
   sentences with a semicolon or a dash.
3. **Use the active voice.** Say who or what does the action. Write "confit
   refuses the query", not "the query is refused".
4. **Use simple tenses.** Use the present, the past and the future. Use the
   imperative for instructions.
5. **Do not remove words to make text shorter.** Keep "the", "a" and "that".
   Do not write in note style ("Gate green, merged, next T5").
6. **Put the condition first.** Write "If the gate fails, do not push", not
   "Do not push if the gate fails".
7. **Keep the strength of a claim.** Write "may" or "about" when the
   evidence is not complete. Write a fact as a fact. Do not stack two
   qualifiers.

### Paragraphs and lists

1. **Write one topic in each paragraph.** The first sentence gives the
   topic. A paragraph has a maximum of six sentences.
2. **Use a numbered list for steps.** Use a bulleted list for three or more
   items of the same type. Use a table to compare items.
3. **Give numbers with their units.** Use metric units (ms, µs, s, MB).
   Say what measured the number, and on which commit.

### Text for the owner

The owner does not see the session. Write each report, PR text and
question so that the owner can read it alone.

1. **Say what a change does, not only its identifier.** Do not write "T3"
   or "#363" alone. Write "shared subexpressions within a call (#363)".

## The glossary

[`GLOSSARY.md`](../../../GLOSSARY.md) holds the terms of this project. Each entry says
what a thing **is**, in one or two sentences. Under _Avoid_ it lists the
words that we do not use for that thing.

1. **Use the glossary term.** If you see a term used in two ways, stop and
   find which meaning is correct. Then make the code, the docs and the
   glossary agree.
2. **Add a term when it becomes fixed,** in the PR that fixes it. Do not
   collect terms at the end of the work.
3. **Keep only terms in the glossary.** Do not put implementation details,
   specifications, decisions or general programming words in it. A decision
   goes to a loop's `decisions/`. A specification goes to the package docs.
4. **If the code and the glossary disagree,** say so in the PR. Then change
   one of them.
5. **Make the glossary shorter when you can.** Remove a term that the
   project does not use now.

## Checks

### The checker

`scripts/prose_check.py` finds sentences of more than 25 words and
semicolons. The pre-commit hook runs it on the files in its
`SCOPE`, so CI fails on these problems. When a doc that the owner reviews or
maintains follows the rules, add it to `SCOPE`.

### The fresh-reader test

1. Run the checker on the text, and fix what it finds:
   `uv run --no-sync python scripts/prose_check.py - < draft.md`
2. Start a new agent with a small model. Give it only the text,
   `GLOSSARY.md`, and this prompt:

   > You are the owner of this repository. You did not see the session that
   > wrote the text below. List each word, name, number, identifier or
   > reference that you cannot understand from the text and the glossary
   > alone. List each sentence that you must read twice. List each word
   > that the glossary puts under _Avoid_ and that the text uses for the
   > concept of that entry. Do not rewrite the text.

3. Fix each item that the reader lists:
   - explain the term where it first occurs;
   - use the glossary term;
   - or remove the item.

   If the item is a new concept, add it to `GLOSSARY.md` first
   ("The glossary" above).
4. Send the text to the owner. Do not send the reader's list.
