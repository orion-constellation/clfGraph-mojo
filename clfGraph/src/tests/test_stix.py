import pytest


def test_parses_valid_stix_json_successfully(self):
        valid_stix_json = '{"type": "bundle", "id": "bundle--1234", "objects": []}'
        result = parse_stix_data(valid_stix_json)
        assert result is not None
        assert result['type'] == 'bundle'

def test_raises_value_error_for_invalid_json_string(self):
        invalid_stix_json = '{"type": "bundle", "id": "bundle--1234", "objects": [}'
        with pytest.raises(ValueError, match="Error parsing STIX data"):
            parse_stix_data(invalid_stix_json)