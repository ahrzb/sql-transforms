# Generate every missing IR opcode

**Done when:** The IR differential and a totality test cover all listed missing opcodes.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

ir::gen::gen_program never emits Dtof, Itod, Dop, Dcast, DcastOk, Dunary, Dtos, StoiOpt, StofOpt, ReMatch, ReExtract, ReReplace, ExternCall, ProbeRange or ProbeRead.

## Wanted

Generate valid operands and contexts for each missing opcode. Add a totality check beside the execution differential.

## Involved

- `packages/confit/src/specializer/ir/`: `gen::gen_program`.
- `packages/confit/src/specializer/exec/tests.rs`: opcode differential and totality.

## Out of scope

Admitting new SQL forms.
