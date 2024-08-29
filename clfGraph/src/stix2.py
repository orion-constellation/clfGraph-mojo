import json

import stix2
import stix2.utils
import stix2patterns

'''
Parse STIX data from a JSON string.

Parameters:
- stix_json (str): A JSON string containing STIX data to be parsed.

Returns:
- bundle: Parsed STIX data in a bundle format.

Raises:
- ValueError: If there is an error parsing the STIX data.
'''
def parse_stix_data(stix_json):
    try:
        bundle = stix2.parse(stix_json, allow_custom=True)
        return bundle
    except Exception as e:
        raise ValueError(f"Error parsing STIX data: {e}")
        pass
    return bundle: json


#@FIXME use Rustworkx to build graph
def build_graph(bundle):
    nodes = {}
    edges = {}
    try:
        ix=0
        for obj in bundle.objects:
            if obj.type != 'relationship':
            nodes.extend(ix(obj))
        else:
            edges.append({
                "source": obj.source_ref,
                "target": obj.target_ref,
                "type": obj.relationship_type
    except Exception as e:
        raise ValueError(f"Error parsing STIX data: {e}", exc_info=True)
            })
            
    return nodes, edges

