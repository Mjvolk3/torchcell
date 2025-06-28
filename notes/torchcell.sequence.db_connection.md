---
id: zg0nj4rl63b8jsdbn3vbjfu
title: Db_connection
desc: ''
updated: 1750979995881
created: 1750979994581
---
# Python Type System: A Comprehensive Guide

## Introduction

Python's type system has evolved significantly, offering powerful tools for writing maintainable, self-documenting code. While Python remains dynamically typed at runtime, type hints enable static type checkers (like mypy, pyright) to catch errors before execution and improve IDE support.

This guide covers essential type system concepts with practical examples, using modern Python 3.10+ syntax.

## TypeVar: Creating Type Variables

### What is TypeVar?

`TypeVar` is a way to create a "type variable" - a placeholder that represents a type to be determined later. Think of it as a generic parameter that maintains type consistency across function signatures.

### Basic TypeVar Forms

```python
from typing import TypeVar

# Unconstrained - can be any type
T = TypeVar('T')

# Constrained - limited to specific types
Number = TypeVar('Number', int, float, complex)

# Bounded - must be a specific type or its subclasses
from collections.abc import Sequence
S = TypeVar('S', bound=Sequence)
```

### Why Use TypeVar?

TypeVar solves a fundamental problem: maintaining type relationships between inputs and outputs.

```python
# Problem: Without TypeVar
def get_first(items: list) -> Any:  # Returns Any - no type safety!
    return items[0]

# Solution: With TypeVar
T = TypeVar('T')

def get_first(items: list[T]) -> T:  # Returns the same type as list contains
    return items[0]

# The type checker now understands:
numbers = [1, 2, 3]
first_num = get_first(numbers)  # Type is int

names = ["Alice", "Bob"]
first_name = get_first(names)   # Type is str
```

### Common Misconception: list[int, str]

Python's `list` type only accepts one type parameter. Here's what's valid:

```python
# ✅ Valid type hints:
list[int]               # List of only integers
list[str]               # List of only strings  
list[int | str]         # List containing EITHER ints OR strings
list[list[int]]         # List of lists of integers

# ❌ Invalid - list doesn't accept multiple parameters:
list[int, str]          # This is NOT valid Python!
```

### TypeVar Preserves Type Information

Here's a practical example showing why TypeVar is superior to alternatives:

```python
from typing import TypeVar

T = TypeVar('T')

# TypeVar maintains precise types
def identity(value: T) -> T:
    """Returns the value unchanged with its exact type preserved"""
    return value

def make_pair(value: T) -> tuple[T, T]:
    """Creates a pair of the same value and type"""
    return (value, value)

def swap(pair: tuple[T, T]) -> tuple[T, T]:
    """Swaps elements in a tuple while preserving types"""
    return (pair[1], pair[0])

# Usage demonstrates type preservation:
x = identity(42)                    # x: int
y = identity("hello")               # y: str
pair_int = make_pair(10)            # pair_int: tuple[int, int]
pair_str = make_pair("test")        # pair_str: tuple[str, str]
swapped = swap(pair_int)            # swapped: tuple[int, int]

# Without TypeVar, you lose precision:
def bad_identity(value: int | str) -> int | str:
    return value

z = bad_identity(42)  # z: int | str (less precise!)
```

## Generic Classes

`Generic[T]` makes a class parameterizable by type. This allows you to create classes that work with different types while maintaining type safety.

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Box(Generic[T]):
    def __init__(self, content: T) -> None:
        self.content = content
    
    def get(self) -> T:
        return self.content

# Usage with type hints:
int_box: Box[int] = Box(42)
str_box: Box[str] = Box("hello")

# Type checker knows:
x = int_box.get()  # x is int
y = str_box.get()  # y is str
```

## Protocols

`Protocol` defines a structural type - any class that has the required methods/attributes matches the protocol, without explicit inheritance.

```python
from typing import Protocol

class DatabaseProtocol(Protocol):
    """Any class with these methods matches this protocol"""
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def execute(self, query: str) -> Any: ...

# These classes match the protocol without inheriting from it:
class SQLiteDB:
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def execute(self, query: str) -> Any: ...

class PostgresDB:
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def execute(self, query: str) -> Any: ...
```

## Type[T] vs type[T]

- `type[T]` (Python 3.9+) or `Type[T]` (older) represents the class itself, not an instance

```python
from typing import Type  # Only needed for Python < 3.9

def create_instance(cls: type[T], *args) -> T:
    """Takes a class and returns an instance of that class"""
    return cls(*args)

# Usage:
my_string = create_instance(str, "hello")  # Returns str instance
my_list = create_instance(list)           # Returns list instance
```

## Putting It All Together: DatabaseConnectionManager

Here's how these concepts work together in our `DatabaseConnectionManager`:

```python
from typing import TypeVar, Generic, Protocol

# 1. Define what a database should look like
class DatabaseProtocol(Protocol):
    """Structural type for database-like objects"""
    pass

# 2. Create a bounded type variable
T = TypeVar('T', bound=DatabaseProtocol)

# 3. Create a generic class parameterized by T
class DatabaseConnectionManager(Generic[T]):
    def __init__(self, db_path: str, db_class: type[T]):
        # db_class is the class itself, not an instance
        self.db_class = db_class
        self.db_path = db_path
    
    def get_connection(self) -> T:
        # Returns an instance of whatever type T is
        return self.db_class(self.db_path)

# 4. Create type-specific aliases
GffutilsConnectionManager = DatabaseConnectionManager[FeatureDB]
```

## Benefits of This Approach

1. **Type Safety**: The type checker knows exactly what type `get_connection()` returns
2. **IDE Support**: Autocomplete works correctly for the specific database type
3. **Documentation**: The types serve as inline documentation
4. **Flexibility**: Can work with any database type that matches the protocol
5. **Reusability**: One implementation works for multiple database types

## Example Usage

```python
# Type checker knows this returns FeatureDB
gff_manager = DatabaseConnectionManager[FeatureDB]("path.db", FeatureDB)
db = gff_manager.get_connection()  # db is FeatureDB

# Could also work with other database types
class CustomDB(DatabaseProtocol):
    def __init__(self, path: str): ...

custom_manager = DatabaseConnectionManager[CustomDB]("path.db", CustomDB)
custom_db = custom_manager.get_connection()  # custom_db is CustomDB
```

## Python Version Compatibility

Since we're using Python 3.10+, we can use all modern syntax:

- ✅ `list[T]`, `dict[K, V]` instead of `List[T]`, `Dict[K, V]`
- ✅ `type[T]` instead of `Type[T]`
- ✅ `X | Y` instead of `Union[X, Y]`
- ✅ `X | None` instead of `Optional[X]`
- ✅ No need to import `Optional`, `Union`, `List`, `Dict`, etc. from typing

Example of modern Python 3.10+ type hints:

```python
# Old (pre-3.9)
from typing import Optional, Union, List, Dict, Type

def process(data: Optional[List[str]], 
           mapping: Dict[str, int],
           cls: Type[MyClass]) -> Union[str, int]:
    ...

# Modern (3.10+)
def process(data: list[str] | None, 
           mapping: dict[str, int],
           cls: type[MyClass]) -> str | int:
    ...
```
