import stix2
import stix2.utils
import stix2patterns


def parse_stix_data(stix_json):
    try:
        bundle = stix2.parse(stix_json, allow_custom=True)
        return bundle
    except Exception as e:
        raise ValueError(f"Error parsing STIX data: {e}")
        pass
    return bundle

def build_graph(bundle):
    nodes = {}
    edges = {}
    try:
        for obj in bundle.objects:
            if obj.type != 'relationship':
            nodes.extend(ix(obj))
        else:
            relationships.append({
                "source": obj.source_ref,
                "target": obj.target_ref,
                "type": obj.relationship_type
    except Exception as e:
        raise ValueError(f"Error parsing STIX data: {e}", exc_info=True)
            })
            
    return nodes, relationships

