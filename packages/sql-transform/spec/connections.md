# Connections and leases

## Caller connections

**claim: caller-connection-execution.** Fit, join probes, and batch execute on the caller connection when supplied.

*Evidence:* `_projection_test.py::test_caller_catalog_probes_and_batch_restore_observed_threads`.

## Leased names

**claim: leased-registrations.** Each execution leases private table and function names on a shared connection.
This includes scalar UDFs, fit functions, learned transform functions, and numbered raw-fit sources.
Function adapters change names without duplicating registration or requiring dataclass UDFs.
Overlapping artifacts therefore keep independent registrations, including outstanding general lazy relations.

*Evidence:* `_lease_test.py::test_release_gives_them_back_on_demand`, `_lifecycle_test.py::test_lazy_retention_is_bounded_by_the_artifact`.

## Cleanup and errors

**claim: cleanup-keeps-first-error.** Cleanup releases only registrations that succeeded.
A partial registration failure releases earlier registrations.
The first execution or callback error remains the reported error.
Cleanup or thread-restoration errors cannot replace it or skip later cleanup.

*Evidence:* `_lease_test.py::test_a_foreign_transforms_own_refusal_reaches_the_caller`.

## The thread setting

**claim: projection-thread-setting.** Projection fit, probes, and batch use `threads=1`.
They save the observed thread setting and restore that value with SET after releasing registrations.
They do not RESET the database setting.
The setting is database-wide.
Callers must not change it concurrently on the same database.
General `SQLTransform` retains its own thread policy.

*Evidence:* `_projection_test.py::test_caller_catalog_probes_and_batch_restore_observed_threads`.
