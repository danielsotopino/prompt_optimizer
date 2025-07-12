"""
YAML input parser utility for the SPO Framework
Handles loading and validation of YAML input files
"""
import yaml
import json
from typing import Any, Dict, List, Union
from pathlib import Path

class YAMLParser:
    """Utility class for parsing YAML input files"""
    
    @staticmethod
    def load_yaml_file(file_path: str) -> Dict[str, Any]:
        """Load YAML file and return parsed content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {file_path}: {e}")
    
    @staticmethod
    def parse_inputs(file_path: str) -> List[str]:
        """Parse inputs from YAML file"""
        data = YAMLParser.load_yaml_file(file_path)
        
        if 'inputs' in data:
            return data['inputs']
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("YAML file must contain 'inputs' key or be a list")
    
    @staticmethod
    def parse_prompt_config(file_path: str) -> Dict[str, Any]:
        """Parse prompt configuration from YAML file"""
        data = YAMLParser.load_yaml_file(file_path)
        
        result = {}
        
        # Parse prompt (can be multiline using YAML |)
        if 'prompt' in data:
            result['prompt'] = data['prompt']
        
        # Parse task description
        if 'task' in data:
            result['task'] = data['task']
            
        # Parse inputs
        if 'inputs' in data:
            result['inputs'] = data['inputs']
            
        # Parse expected outputs
        if 'expected' in data:
            result['expected'] = data['expected']
            
        # Parse other configuration
        for key in ['iterations', 'model', 'strategies', 'metrics']:
            if key in data:
                result[key] = data[key]
        
        return result
    
    @staticmethod
    def detect_file_format(file_path: str) -> str:
        """Detect if file is JSON or YAML based on extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in ['.yaml', '.yml']:
            return 'yaml'
        elif extension == '.json':
            return 'json'
        else:
            # Try to parse as YAML first, fallback to JSON
            try:
                with open(file_path, 'r') as f:
                    yaml.safe_load(f)
                return 'yaml'
            except:
                return 'json'
    
    @staticmethod
    def load_inputs_flexible(file_path: str) -> List[str]:
        """Load inputs from either JSON or YAML file"""
        file_format = YAMLParser.detect_file_format(file_path)
        
        if file_format == 'yaml':
            data = YAMLParser.load_yaml_file(file_path)
            
            # Check for new data format with input/expected pairs
            if 'data' in data and isinstance(data['data'], list):
                # Support both 'message'/'input' field names
                return [item.get('message', item.get('input', '')) for item in data['data'] 
                       if 'message' in item or 'input' in item]
            # Check for inputs key
            elif 'inputs' in data:
                return data['inputs']
            # Check if it's a direct list
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("YAML file must contain 'inputs', 'data', or be a list")
        else:
            # Load JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif 'inputs' in data:
                    return data['inputs']
                else:
                    raise ValueError("JSON file must be a list or contain 'inputs' key")
    
    @staticmethod
    def load_expected_flexible(file_path: str) -> List[str]:
        """Load expected outputs from either JSON or YAML file"""
        file_format = YAMLParser.detect_file_format(file_path)
        
        if file_format == 'yaml':
            data = YAMLParser.load_yaml_file(file_path)
            
            # Check for new data format with input/expected pairs
            if 'data' in data and isinstance(data['data'], list):
                # Support both 'classification'/'expected' field names
                return [item.get('classification', item.get('expected', '')) for item in data['data'] 
                       if 'classification' in item or 'expected' in item]
            # Check for expected key
            elif 'expected' in data:
                return data['expected']
            # Check if it's a direct list
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("YAML file must contain 'expected', 'data', or be a list")
        else:
            # Load JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif 'expected' in data:
                    return data['expected']
                else:
                    raise ValueError("JSON file must be a list or contain 'expected' key")
    
    @staticmethod
    def save_results_yaml(data: Dict[str, Any], file_path: str) -> None:
        """Save results to YAML file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving YAML file {file_path}: {e}")
    
    @staticmethod
    def should_save_as_yaml(file_path: str) -> bool:
        """Check if output should be saved as YAML based on file extension"""
        return file_path.lower().endswith(('.yaml', '.yml'))