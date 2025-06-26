from __future__ import with_statement, division, print_function
import os
from register_virtual_stack import Register_Virtual_Stack_MT as RVSMT
from ij import IJ, WindowManager

#@String src
#@String out
#@String output
#@String ref
#@boolean shrink

if not src.endswith("/"):  src += "/"
if not out.endswith("/"):  out += "/"
if not os.path.isdir(out): os.makedirs(out)

print(">>> Register_Virtual_Stack_MT.exec")
print("    source :", src)
print("    target :", out)
print("    ref    :", ref or "(auto)")
print("    shrink :", shrink)

tif_files = sorted(f for f in os.listdir(src) if f.endswith(".tiff"))
ref = tif_files[len(tif_files)//2]

p = RVSMT.Param()
p.featuresModelIndex      = RVSMT.RIGID
p.registrationModelIndex  = RVSMT.RIGID
p.sift.maxOctaveSize      = 1024
p.minInlierRatio          = 0.05

RVSMT.exec(src, out, out, ref, p, shrink)


tif_list = sorted(f for f in os.listdir(out) if f.lower().endswith(('.tif', '.tiff')))
if len(tif_list) < 2:
    IJ.log("No registered TIFFs found in " + out)
    from java.lang import System
    System.exit(0)

for f in tif_list:
    IJ.open(out + f)

IJ.run("Images to Stack", "name=Stack use")
stack_imp = WindowManager.getImage("Stack")
save_path = output
IJ.saveAs(stack_imp, "Tiff", save_path)

IJ.run("Close All")
from java.lang import System
System.exit(0) 
