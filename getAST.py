from src.conf import TORCH_BASE
import ast
import click
from loguru import logger
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any

class ModuleAnalyzer:
    # 分析范围限制
    analysis_scope: Optional[Path] = None
    # 已分析的文件
    analyzed_files: Set[Path] = set()
    # 模块根目录
    module_root: Path
    # 要分析的函数
    target_function: Optional[str] = None
    # 目标函数所在文件
    target_file: Optional[Path] = None
    # 符号位置映射 {符号全称: (开始行, 结束行)}
    symbol_positions: Dict[str, Tuple[int, int]] = {}
    # 调用关系映射 {调用者全称: [被调用者全称列表]}
    call_graph: Dict[str, List[str]] = defaultdict(list)
    # 模块名到文件路径的映射
    module_map: Dict[str, str] = {}
    
    def __init__(self, module_root: Path, target_file: Path, target_function: str, analysis_scope: Optional[Path] = None):
        self.module_root = module_root
        self.target_file = target_file
        self.target_function = target_function
        self.analysis_scope = analysis_scope
        self.analyzed_files = set()
        self.symbol_positions = {}
        self.call_graph = defaultdict(list)
        self.module_map = {}
    
    def is_parent_path(self, parent: Path, child: Path) -> bool:
        """Check if parent path is a parent directory of child path."""
        try:
            # Convert both paths to absolute paths
            parent = parent.resolve()
            child = child.resolve()
            # Check if child path is relative to parent path
            return child.is_relative_to(parent)
        except ValueError:
            return False
    
    def find_file_by_abs_m(self, module_name: str) -> Optional[Path]:
        """absolute path like torch._inductor"""
        module_name_list = module_name.split('.')
        assert len(module_name_list) > 0
        module_name_list[-1] = module_name_list[-1] + '.py'
        module_file_path = Path(self.module_root).joinpath(*module_name_list)
        if module_file_path.exists():
            return module_file_path
        else:
            return None
    
    def find_file_by_relative_m(self, module_name: str, current_file: Path) -> Optional[Path]:
        """relative path like ._inductor"""
        module_name_list = module_name.split('.')
        assert len(module_name_list) > 0
        module_name_list[-1] = module_name_list[-1] + '.py'
        module_file_path = current_file.parent.joinpath(*module_name_list)
        if module_file_path.exists():
            return module_file_path
        else:
            logger.warning(f"Could not find relative module file for: {module_name}")
            return None
    
    def find_file_by_m_name(self, module_name: str, current_file: Path) -> Optional[Path]:
        """find the file by module name"""
        if module_name.startswith('.'):
            return self.find_file_by_relative_m(module_name, current_file)
        else:
            return self.find_file_by_abs_m(module_name)
    
    def _get_module_name(self, file_path: str) -> str:
        """从文件路径获取模块名"""
        try:
            rel_path = Path(file_path).relative_to(self.module_root)
            module_parts = list(rel_path.parent.parts)
            if rel_path.stem != "__init__":
                module_parts.append(rel_path.stem)
            return ".".join(module_parts)
        except Exception:
            return "unknown"
    
    def _find_module_file(self, module_name: str, current_file: Path) -> Optional[Path]:
        """查找模块对应的文件"""
        # 检查是否已经在映射中
        if module_name in self.module_map:
            return Path(self.module_map[module_name])
        
        # 尝试查找模块文件
        if module_name.startswith('.'):
            # 相对导入
            module_file = self.find_file_by_relative_m(module_name, current_file)
        else:
            # 绝对导入
            module_file = self.find_file_by_abs_m(module_name)
        
        if module_file:
            self.module_map[module_name] = str(module_file)
            return module_file
        return None
    
    def analyze_file(self, entry_file: Path):
        """Analyze a single Python file."""
        # Skip if already analyzed
        if entry_file in self.analyzed_files:
            logger.debug(f"Skipping already analyzed file: {entry_file}")
            return
        
        # Check if file is within analysis scope
        if self.analysis_scope and not self.is_parent_path(self.analysis_scope, entry_file):
            logger.debug(f"Skipping file outside analysis scope: {entry_file}")
            return
        
        logger.info(f"Analyzing file: {entry_file}")
        self.analyzed_files.add(entry_file)
        
        try:
            with open(entry_file, "r", encoding="utf-8") as f:
                code = f.read()
            
            tree = ast.parse(code)
            analyzer = ASTVisitor(str(entry_file), self._get_module_name(str(entry_file)))
            analyzer.visit(tree)
            
            # 处理导入语句
            for imp in analyzer.imports:
                if imp["type"] == "import":
                    module_name = imp["name"]
                    
                    # 查找并分析导入的模块
                    module_file = self._find_module_file(module_name, entry_file)
                    if module_file:
                        if module_file not in self.analyzed_files:
                            self.analyze_file(module_file)
                else:  # from import
                    module_name = imp["module"] if imp["module"] else ""
                    if not module_name:
                        continue
                    
                    # 查找并分析导入的模块
                    module_file = self._find_module_file(module_name, entry_file)
                    if module_file:
                        if module_file not in self.analyzed_files:
                            self.analyze_file(module_file)
            
            # 处理函数和类信息
            for symbol_name, (start_line, end_line) in analyzer.symbol_positions.items():
                # 使用模块名+符号名作为全称
                full_symbol_name = f"{analyzer.module_name}.{symbol_name}"
                self.symbol_positions[full_symbol_name] = (start_line, end_line)
            
            # 处理调用关系
            for caller, callees in analyzer.call_graph.items():
                # 使用模块名+调用者名作为全称
                full_caller_name = f"{analyzer.module_name}.{caller}"
                
                # 处理每个被调用者
                for callee in callees:
                    # 检查是否是目标函数的调用
                    if self.target_function and caller != self.target_function:
                        continue
                    
                    # 获取被调用者信息
                    if "." in callee:
                        # 模块调用
                        module_parts = callee.split(".")
                        module_name = ".".join(module_parts[:-1])
                        func_name = module_parts[-1]
                        
                        # 查找模块文件
                        module_file = self._find_module_file(module_name, entry_file)
                        if module_file:
                            callee_module = self._get_module_name(str(module_file))
                            full_callee_name = f"{callee_module}.{func_name}"
                        else:
                            full_callee_name = callee
                    else:
                        # 本地函数调用
                        full_callee_name = f"{analyzer.module_name}.{callee}"
                    
                    # 添加到调用图
                    self.call_graph[full_caller_name].append(full_callee_name)
                    
                    # 如果是目标函数的调用，递归分析被调用函数
                    if self.target_function and caller == self.target_function:
                        if "." in callee:
                            module_parts = callee.split(".")
                            module_name = ".".join(module_parts[:-1])
                            module_file = self._find_module_file(module_name, entry_file)
                            if module_file and module_file not in self.analyzed_files:
                                self.analyze_file(module_file)
            
        except Exception as e:
            logger.error(f"Error analyzing file {entry_file}: {str(e)}")
            raise e
    
    def print_results(self):
        """Print analysis results."""
        print(f"\nFunction Call Graph Analysis Results:")
        print(f"Target function: {self.target_function} in {self.target_file}")
        if self.analysis_scope:
            print(f"Analysis scope: {self.analysis_scope}")
        print(f"Total files analyzed: {len(self.analyzed_files)}")
        
        print("\nSymbol Positions:")
        # 打印所有符号的位置信息
        for symbol_name, (start_line, end_line) in sorted(self.symbol_positions.items()):
            print(f"\n{symbol_name}:")
            print(f"  Lines: {start_line}-{end_line}")
        
        print("\nCall Graph:")
        # 按调用者分组显示调用图
        for caller, callees in sorted(self.call_graph.items()):
            print(f"\n{caller} calls:")
            for callee in callees:
                print(f"  - {callee}")
        
            
    def get_callchain_by_func(self, function_name: str) -> List[str]:
        """获取指定函数的调用关系"""
        return self.call_graph[function_name]

class ASTVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, module_name: str):
        self.file_path = file_path
        self.module_name = module_name
        # 符号位置映射 {符号名: (开始行, 结束行)}
        self.symbol_positions: Dict[str, Tuple[int, int]] = {}
        # 调用关系映射 {调用者: [被调用者列表]}
        self.call_graph: Dict[str, List[str]] = defaultdict(list)
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None
        self.imports: List[Dict[str, Any]] = []
        self.module_aliases: Dict[str, str] = {}  # 记录模块别名
    
    def visit_Import(self, node: ast.Import) -> None:
        # Handle regular import statements (e.g., import os)
        for alias in node.names:
            import_info: Dict[str, Any] = {
                "type": "import",
                "name": alias.name,
                "module": None,
                "asname": alias.asname,
                "line": node.lineno,
                "col": node.col_offset
            }
            self.imports.append(import_info)
            # 记录模块别名
            self.module_aliases[alias.asname or alias.name] = alias.name
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Handle from ... import statements (e.g., from os import path)
        module = node.module if node.module else ""
        for alias in node.names:
            import_info: Dict[str, Any] = {
                "type": "from",
                "module": module,
                "name": alias.name,
                "asname": alias.asname,
                "line": node.lineno,
                "col": node.col_offset
            }
            self.imports.append(import_info)
            # 记录从模块导入的函数
            if module:
                self.module_aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        if self.current_function is None:
            return
            
        # Get the name of the called function/class
        if isinstance(node.func, ast.Name):
            # 直接函数调用
            called_name = node.func.id
            # 检查是否是导入的函数
            if called_name in self.module_aliases:
                called_name = self.module_aliases[called_name]
        elif isinstance(node.func, ast.Attribute):
            # 处理模块调用 (e.g., torch.tensor, self.method())
            parts: List[str] = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                # 反转以获得正确的顺序 (e.g., torch.nn.functional)
                parts.reverse()
                # 检查模块别名
                if parts[0] in self.module_aliases:
                    parts[0] = self.module_aliases[parts[0]]
                called_name = ".".join(parts)
            else:
                # Skip complex attribute access for now
                return
        else:
            return
            
        # 过滤掉内置函数调用
        if called_name in dir(__builtins__):
            return
            
        # Add to call graph
        self.call_graph[self.current_function].append(called_name)
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # 获取函数名和行号范围
        start_line = node.lineno
        end_line = self._get_end_line(node)
        
        # 存储函数信息
        if self.current_class:
            # 类方法
            symbol_name = f"{self.current_class}.{node.name}"
        else:
            # 普通函数
            symbol_name = node.name
        
        self.symbol_positions[symbol_name] = (start_line, end_line)
        
        # Track the current function for call relationship analysis
        prev_function = self.current_function
        self.current_function = symbol_name
        
        # Visit the function body to analyze calls
        self.generic_visit(node)
        
        # Restore the previous function context
        self.current_function = prev_function
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # 获取类名和行号范围
        start_line = node.lineno
        end_line = self._get_end_line(node)
        
        # 存储类信息
        self.symbol_positions[node.name] = (start_line, end_line)
        
        # Track the current class for method analysis
        prev_class = self.current_class
        self.current_class = node.name
        
        # Visit the class body
        self.generic_visit(node)
        
        # Restore the previous class context
        self.current_class = prev_class
    
    def _get_end_line(self, node: ast.AST) -> int:
        """获取节点的结束行号"""
        # 使用 lineno 作为基础行号
        base_line = getattr(node, 'lineno', 0)
        
        # 遍历所有子节点找到最大的行号
        max_line = base_line
        for child in ast.iter_child_nodes(node):
            child_line = getattr(child, 'lineno', 0)
            max_line = max(max_line, child_line)
        
        return max_line



def module_name_to_file_path(module_name: str, base_path: Path) -> Path:
    """将模块名转换为文件路径"""
    module_name_list = module_name.split('.')
    assert len(module_name_list) > 0
    module_name_list = module_name_list[:-1] # remove symbol name
    module_name_list[-1] = module_name_list[-1] + '.py'
    module_file_path = base_path.joinpath(*module_name_list)
    return module_file_path

@click.command()
@click.argument('entry_file', type=click.Path(exists=True), required=True)
@click.argument('target_function', type=str, required=True)
@click.option('--module-root', type=click.Path(exists=True), required=True, help='The root directory of the module')
@click.option('--analysis-scope', type=click.Path(exists=True), help='Limit analysis to files within this scope (absolute path)')
def main(entry_file: str, target_function: str, module_root: str, analysis_scope: Optional[str] = None) -> None:
    """Analyze the call graph of a specific function in a Python file."""
    # Convert string paths to Path objects
    entry_path = Path(entry_file)
    module_root_path = Path(module_root)
    analysis_scope_path = Path(analysis_scope) if analysis_scope else None
    
    analyzer = ModuleAnalyzer(module_root_path, entry_path, target_function, analysis_scope_path)
    analyzer.analyze_file(entry_path)
    
    callchain:List[str] = ["torch._inductor.aoti_eager.extract_tensor_list_metadata"]
    for level in range(2):
        new_callchain:List[str] = []
        for func in callchain:
            file_path = module_name_to_file_path(func, TORCH_BASE)
            analyzer.analyze_file(file_path)
            new_callchain.extend(analyzer.get_callchain_by_func(func))
        callchain = new_callchain
        logger.info(f"Level {level} callchain: {callchain}")
    print(callchain)
    

if __name__ == "__main__":
    main()
