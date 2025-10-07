# save as: ckpt_inspect.py
import sys, torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
state = ckpt.get("model") or ckpt.get("state_dict") or ckpt

def shp(k):
    t = state.get(k)
    return None if t is None else tuple(t.shape)

print("=== ARGS (if saved) ===")
print(ckpt.get("args", {}))

print("\n=== SPEC adapter ===")
print("spec.proj.0.weight:", shp("spec.proj.0.weight"))
print("spec.proj.1.weight:", shp("spec.proj.1.weight"))

print("\n=== MOL adapter ===")
print("mol.proj.0.weight :", shp("mol.proj.0.weight"))
print("mol.proj.1.weight :", shp("mol.proj.1.weight"))

print("\n=== Mapper (mu branch) ===")
print("mapB.mu.0.weight  :", shp("mapB.mu.0.weight"))
print("mapB.mu.2.weight  :", shp("mapB.mu.2.weight"))

print("\n=== Mapper (lv branch, if Gaussian) ===")
print("mapB.lv.0.weight  :", shp("mapB.lv.0.weight"))
print("mapB.lv.2.weight  :", shp("mapB.lv.2.weight"))
