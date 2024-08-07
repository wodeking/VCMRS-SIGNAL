# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
import os
import importlib.util
import vcmrs
from . import component_bypass


def load_component(component, component_prefix, name, ctx):
  ''' Load a component with a specific name
  '''
  # bypass component
  if name.lower() == 'bypass':
    return component_bypass.Bypass(ctx)

  vcmrs_dir = os.path.dirname(vcmrs.__file__)
  component_fname = os.path.join(vcmrs_dir, component, f"{component_prefix}_{name.lower()}.py")
  spec = importlib.util.spec_from_file_location(f"vcmrs.{component}.{name}", component_fname)

  foo = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(foo)
  component = getattr(foo, name)(ctx)
  return component


