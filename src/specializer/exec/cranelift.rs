//! Cranelift-jit backend skeleton: smoke-level only for now — proves the
//! toolchain JITs on this platform (spike re-validated in-tree). The real
//! per-row backend lands next: extern "C" fn(ctx) -> i64 ABI per the
//! TASK-44 stretch plan.

#[cfg(test)]
mod tests {
    use cranelift_codegen::ir::{types, AbiParam, InstBuilder};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    #[test]
    fn jit_smoke_add_two_i64() {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").unwrap();
        flags.set("is_pic", "false").unwrap();
        let isa = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
            .unwrap()
            .finish(settings::Flags::new(flags))
            .unwrap();
        let mut module = JITModule::new(JITBuilder::with_isa(
            isa,
            cranelift_module::default_libcall_names(),
        ));

        let mut ctx = module.make_context();
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let (x, y) = (b.block_params(entry)[0], b.block_params(entry)[1]);
        let sum = b.ins().iadd(x, y);
        b.ins().return_(&[sum]);
        b.finalize();

        let id = module
            .declare_function("add", Linkage::Export, &ctx.func.signature)
            .unwrap();
        module.define_function(id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);
        module.finalize_definitions().unwrap();

        let code = module.get_finalized_function(id);
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(40, 2), 42);
    }
}
