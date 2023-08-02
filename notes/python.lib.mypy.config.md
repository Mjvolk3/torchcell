---
id: t4a99ijrtskiyscwoasuziu
title: Config
desc: ''
updated: 1690582523121
created: 1690582088477
---
## VsCode MyPy Extension

Sure, let's go through these:

1. **ignore_missing_imports = true:** This will silence any errors about importing modules that MyPy can't find type annotations for.

2. **show_error_codes = true:** This will show error codes in error messages. This can be useful for quickly identifying the types of errors that occur, especially when used in conjunction with `--explain` command-line option.

3. **exclude = "(build|data|dist|docs/src|images|logo|logs|output)/":** This setting tells MyPy to exclude these directories from type-checking. The value is a regular expression pattern that matches directories to exclude.

4. **warn_unused_configs = true:** If a config option in the configuration file has no effect, a warning will be issued. This d occurs when a configuration file option is misspelled.

5. **disallow_any_generics = true:** When this option is enabled, usage of generic types without explicit parameters (like `List` instead of `List[str]`) is prohibited.

6. **disallow_subclassing_any = true:** This disallows subclassing values of type `Any`.

7. **disallow_untyped_calls = true:** When enabled, this option will prohibit calling functions without type annotations from a function with type annotations.

8. **disallow_untyped_defs = true:** As previously mentioned, this option requires that every function definition has type annotations for its arguments and return value.

9. **disallow_incomplete_defs = true:** This prohibits defining a function with some of the arguments or the return value left untyped.

10. **check_untyped_defs = true:** This option allows for the internal bodies of functions without type annotations to be checked.

11. **disallow_untyped_decorators = true:** This prohibits applying function decorators if the decorated function is not fully annotated.

12. **no_implicit_optional = true:** This means that MyPy will treat `None` as a different type that isn't compatible with non-optional types.

13. **warn_redundant_casts = true:** As mentioned before, this raises a warning if you have a type cast that isn't necessary because MyPy can already infer the type.

14. **warn_unused_ignores = true:** Raises a warning if you have a `# type: ignore` comment that isn't actually ignoring any type errors.

15. **warn_return_any = true:** When enabled, a warning is emitted if a function annotated with a non-`Any` return type returns `Any`.

16. **no_implicit_reexport = true:** When true, re-exported names (i.e., names imported into a module and then made available as part of that module's public API) will be treated as if they had been defined in the module where they are re-exported.

17. **strict_equality = true:** When true, MyPy will disallow comparisons between incompatible types with `==` and `!=`.

In general, these settings make MyPy stricter and more precise, forcing you to annotate types more explicitly, and catching more potential errors. They can make your code safer and easier to understand at the expense of requiring more effort to add and maintain type annotations.