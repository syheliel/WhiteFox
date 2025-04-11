import ast
import os
import sys
import click
from loguru import logger
from collections import defaultdict
class FunctionAPIVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}
        self.function_stack = []  # Stack to track nested functions
        self.imports = []
    
    def visit_FunctionDef(self, node):
        # Push current function to stack
        self.function_stack.append(node.name)
        
        # Initialize function info if it's not already initialized
        if node.name not in self.functions:
            self.functions[node.name] = {
                "line": node.lineno,
                "col": node.col_offset,
                "api_calls": []
            }
        
        # Continue visiting the function body
        self.generic_visit(node)
        
        # Pop the current function from stack
        self.function_stack.pop()
    
    def visit_Call(self, node):
        # Only process API calls if we're inside a function
        if self.function_stack:  # Check if we're inside any function
            current_function = self.function_stack[-1]  # Get the innermost function
            
            # Build the full name of the API call
            if isinstance(node.func, ast.Attribute):
                parts = []
                current = node.func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                api_call = ".".join(reversed(parts))
            elif isinstance(node.func, ast.Name):
                api_call = node.func.id
            else:
                # Skip other types of calls
                return
            
            # Add the API call to the current function
            self.functions[current_function]["api_calls"].append({
                "api_call": api_call,
                "line": node.lineno,
                "col": node.col_offset
            })
        
        self.generic_visit(node)

    def visit_Import(self, node):
        # Handle regular import statements (e.g., import os)
        for alias in node.names:
            self.imports.append({
                "type": "import",
                "name": alias.name,
                "asname": alias.asname,
                "line": node.lineno,
                "col": node.col_offset
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # Handle from ... import statements (e.g., from os import path)
        module = node.module if node.module else ""
        for alias in node.names:
            self.imports.append({
                "type": "from",
                "module": module,
                "name": alias.name,
                "asname": alias.asname,
                "line": node.lineno,
                "col": node.col_offset
            })
        self.generic_visit(node)

def extract_function_apis(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        tree = ast.parse(code)
        visitor = FunctionAPIVisitor()
        visitor.visit(tree)
        
        return {
            "functions": visitor.functions,
            "imports": visitor.imports
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise e
        return {"functions": {}, "imports": []}

@click.command()
@click.argument('file_path', type=click.Path(exists=True), default="getAST.py")
def main(file_path):
    """Analyze Python functions, their API calls, and imports in the specified file."""
    result = extract_function_apis(file_path)
    
    if result["imports"]:
        print(f"\nImports found in {file_path}:")
        for imp in result["imports"]:
            if imp["type"] == "import":
                print(f"  - Line {imp['line']}, Column {imp['col']}: import {imp['name']}" + 
                      (f" as {imp['asname']}" if imp['asname'] else ""))
            else:  # from import
                print(f"  - Line {imp['line']}, Column {imp['col']}: from {imp['module']} import {imp['name']}" +
                      (f" as {imp['asname']}" if imp['asname'] else ""))
    
    if result["functions"]:
        print(f"\nFunctions and their API calls found in {file_path}:")
        for func_name, func_info in result["functions"].items():
            print(f"\nFunction: {func_name} (Line {func_info['line']}, Column {func_info['col']})")
            
            if func_info["api_calls"]:
                print("  API Calls:")
                for api_call in func_info["api_calls"]:
                    print(f"    - Line {api_call['line']}, Column {api_call['col']}: {api_call['api_call']}")
            else:
                print("  No API calls")
    else:
        print(f"No functions found in {file_path}")

if __name__ == "__main__":
    main()
