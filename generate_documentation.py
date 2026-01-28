#!/usr/bin/env python3
"""
Documentation Generator for Custom Loss Trainers Module

This script automatically generates documentation from the custom_loss_trainers module
by extracting docstrings, function signatures, and class information.
"""

import inspect
import ast
import os
import sys
from pathlib import Path

def extract_function_info(module_path):
    """
    Extract function and class information from a Python module.
    
    Parameters
    ----------
    module_path : str
        Path to the Python module file
        
    Returns
    -------
    dict
        Dictionary containing function and class information
    """
    with open(module_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    functions = []
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Extract function information
            func_info = {
                'name': node.name,
                'lineno': node.lineno,
                'docstring': ast.get_docstring(node),
                'args': [arg.arg for arg in node.args.args],
                'defaults': [ast.unparse(default) for default in node.args.defaults] if node.args.defaults else [],
                'type': 'function'
            }
            functions.append(func_info)
            
        elif isinstance(node, ast.ClassDef):
            # Extract class information
            class_info = {
                'name': node.name,
                'lineno': node.lineno,
                'docstring': ast.get_docstring(node),
                'methods': [],
                'type': 'class'
            }
            
            # Extract methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info = {
                        'name': item.name,
                        'docstring': ast.get_docstring(item),
                        'args': [arg.arg for arg in item.args.args],
                        'defaults': [ast.unparse(default) for default in item.args.defaults] if item.args.defaults else []
                    }
                    class_info['methods'].append(method_info)
            
            classes.append(class_info)
    
    return {'functions': functions, 'classes': classes}

def generate_markdown_documentation(module_info, output_file):
    """
    Generate markdown documentation from module information.
    
    Parameters
    ----------
    module_info : dict
        Dictionary containing function and class information
    output_file : str
        Path to output markdown file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Custom Loss Trainers Module - Auto-Generated Documentation\n\n")
        f.write("This documentation was automatically generated from the `custom_loss_trainers` module.\n\n")
        
        # Functions section
        f.write("## Functions\n\n")
        for func in module_info['functions']:
            f.write(f"### {func['name']}\n\n")
            
            if func['docstring']:
                f.write(f"{func['docstring']}\n\n")
            
            # Function signature
            args_str = ", ".join(func['args'])
            f.write(f"**Signature:** `{func['name']}({args_str})`\n\n")
            
            f.write("---\n\n")
        
        # Classes section
        f.write("## Classes\n\n")
        for cls in module_info['classes']:
            f.write(f"### {cls['name']}\n\n")
            
            if cls['docstring']:
                f.write(f"{cls['docstring']}\n\n")
            
            # Methods
            if cls['methods']:
                f.write("**Methods:**\n\n")
                for method in cls['methods']:
                    f.write(f"- `{method['name']}({', '.join(method['args'])})`\n")
                    if method['docstring']:
                        f.write(f"  - {method['docstring'].split('.')[0]}.\n")
                f.write("\n")
            
            f.write("---\n\n")

def generate_rst_documentation(module_info, output_file):
    """
    Generate reStructuredText documentation from module information.
    
    Parameters
    ----------
    module_info : dict
        Dictionary containing function and class information
    output_file : str
        Path to output RST file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Custom Loss Trainers Module\n")
        f.write("==========================\n\n")
        f.write("This documentation was automatically generated from the `custom_loss_trainers` module.\n\n")
        
        # Functions section
        f.write("Functions\n")
        f.write("---------\n\n")
        for func in module_info['functions']:
            f.write(f".. function:: {func['name']}({', '.join(func['args'])})\n\n")
            
            if func['docstring']:
                f.write(f"   {func['docstring']}\n\n")
            
            f.write("   .. note::\n")
            f.write(f"      Line {func['lineno']}\n\n")
        
        # Classes section
        f.write("Classes\n")
        f.write("-------\n\n")
        for cls in module_info['classes']:
            f.write(f".. class:: {cls['name']}\n\n")
            
            if cls['docstring']:
                f.write(f"   {cls['docstring']}\n\n")
            
            # Methods
            if cls['methods']:
                f.write("   **Methods:**\n\n")
                for method in cls['methods']:
                    f.write(f"   .. method:: {method['name']}({', '.join(method['args'])})\n\n")
                    if method['docstring']:
                        f.write(f"      {method['docstring']}\n\n")

def main():
    """Main function to generate documentation."""
    # Add the Modules directory to the path
    sys.path.append('Modules')
    
    module_path = 'Modules/custom_loss_trainers.py'
    
    if not os.path.exists(module_path):
        print(f"Error: Module file {module_path} not found.")
        return
    
    print(f"Extracting information from {module_path}...")
    module_info = extract_function_info(module_path)
    
    print(f"Found {len(module_info['functions'])} functions and {len(module_info['classes'])} classes.")
    
    # Generate markdown documentation
    markdown_file = 'custom_loss_trainers_auto_docs.md'
    generate_markdown_documentation(module_info, markdown_file)
    print(f"Generated markdown documentation: {markdown_file}")
    
    # Generate RST documentation
    rst_file = 'custom_loss_trainers_auto_docs.rst'
    generate_rst_documentation(module_info, rst_file)
    print(f"Generated RST documentation: {rst_file}")
    
    # Print summary
    print("\nDocumentation Summary:")
    print("=====================")
    print(f"Functions documented: {len(module_info['functions'])}")
    print(f"Classes documented: {len(module_info['classes'])}")
    
    # List all functions and classes
    print("\nFunctions:")
    for func in module_info['functions']:
        print(f"  - {func['name']}")
    
    print("\nClasses:")
    for cls in module_info['classes']:
        print(f"  - {cls['name']} ({len(cls['methods'])} methods)")

if __name__ == "__main__":
    main() 