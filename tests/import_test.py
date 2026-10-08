import importlib, os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(root))
m = importlib.import_module(os.path.basename(root))
for name, cls in m.NODE_CLASS_MAPPINGS.items():
    it = cls.INPUT_TYPES()
    assert "required" in it and hasattr(cls, cls.FUNCTION) and cls.RETURN_TYPES
    print(name, list(it["required"]))
chunk_text = importlib.import_module(os.path.basename(root) + ".rumik_core").chunk_text
print(chunk_text("यह पहला वाक्य है। " * 30, "steady"), )
print("OK", len(m.NODE_CLASS_MAPPINGS))
