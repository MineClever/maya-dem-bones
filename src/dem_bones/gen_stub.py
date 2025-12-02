# coding=utf-8
import os
import inspect

import mypy.stubgenc

def build_sig_generators():
    gens = []

    for name in (
        "DocstringSignatureGenerator",
        "RstSignatureGenerator",
        "RuntimeSignatureGenerator",
        "SimpleSignatureGenerator",
    ):
        gen_cls = getattr(mypy.stubgenc, name, None)
        if gen_cls is not None:
            try:
                gens.append(gen_cls())
            except Exception:
                pass

    if hasattr(mypy.stubgenc, "FallbackSignatureGenerator"):
        gens.append(mypy.stubgenc.FallbackSignatureGenerator())

    return gens

def generate_stub(module, stubpath):
    sig = inspect.signature(mypy.stubgenc.generate_stub_for_c_module)
    params = sig.parameters

    kwargs = {}

    if "sig_generators" in params:
        kwargs["sig_generators"] = build_sig_generators()

    if "include_docstrings" in params:
        kwargs["include_docstrings"] = True

    mypy.stubgenc.generate_stub_for_c_module(
        module, stubpath, [], **kwargs
    )

def main():
    stubs_folder = os.path.dirname(__file__)
    modules = ["dem_bones._core"]

    errors = []
    for module in modules:
        stubpath = os.path.join(stubs_folder, f"{module}.pyi")

        if os.path.exists(stubpath):
            os.chmod(stubpath, 0o600)
        else:
            with open(stubpath, "x"):
                print(f"create placeholder stub to [{stubpath}]")

        try:
            generate_stub(module, stubpath)
        except Exception as e:
            print(f"Failed:     {stubpath} ({e!r})")
            errors.append(module)
        else:
            print(f"Generated : {stubpath}")

    if errors:
        raise RuntimeError(f"Stub generation failed for {', '.join(errors)}")

if __name__ == "__main__":
    main()
