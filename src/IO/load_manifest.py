import re
import yaml
import argparse
import pprint

def load_yaml_file(filepath):
    """Loads a YAML file and returns the content as a dictionary."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def resolve_placeholders(value, variables):
    """
    Resolves ${...} placeholders in a string using a variables dictionary.
    Supports nested keys like ${datasets.opencap_root}.
    """
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replacer(match):
        key = match.group(1)
        keys = key.split('.')
        val = variables
        for k in keys:
            val = val.get(k)
            if val is None:
                raise KeyError(f"Placeholder '{key}' could not be resolved.")
        return val

    return pattern.sub(replacer, value)

def recursive_resolve(obj, variables):
    """
    Recursively resolves placeholders in a nested structure.
    """
    if isinstance(obj, str):
        return resolve_placeholders(obj, variables)
    elif isinstance(obj, dict):
        return {k: recursive_resolve(v, variables) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_resolve(item, variables) for item in obj]
    return obj

def load_manifest(manifest_path, paths_path):
    """
    Loads and resolves a manifest.yaml using a companion paths.yaml.

    Args:
        manifest_path (str): Path to the manifest YAML file.
        paths_path (str): Path to the paths YAML file.

    Returns:
        dict: Fully resolved manifest dictionary.
    """
    # Load the global path config
    paths_config = load_yaml_file(paths_path)

    # Initial variable context
    context = {
        'datasets': {
            'opencap_root': paths_config['datasets']['opencap_root']
        },
        'outputs_root': paths_config.get('outputs_root')
    }

    # Load the manifest as-is
    manifest_raw = load_yaml_file(manifest_path)

    # Add resolved 'paths.root' to context
    resolved_root = resolve_placeholders(manifest_raw['paths']['root'], context)
    context['paths'] = {'root': resolved_root}

    # Recursively resolve the entire manifest
    manifest_resolved = recursive_resolve(manifest_raw, context)

    return manifest_resolved

def main():
    parser = argparse.ArgumentParser(description="Load and resolve a manifest.yaml file.")
    parser.add_argument('--manifest', '-m', type=str, required=True, help="Path to manifest.yaml")
    parser.add_argument('--paths', '-p', type=str, required=True, help="Path to paths.yaml")
    parser.add_argument('--print', action='store_true', help="Pretty-print the resolved manifest")
    
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest, args.paths)
        if args.print:
            pprint.pprint(manifest)
        else:
            print("Manifest loaded successfully.")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
