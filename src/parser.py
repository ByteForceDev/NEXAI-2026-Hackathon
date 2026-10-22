"""
Parser for standard Solomon Vehicle Routing Problem with Time Windows (VRPTW) benchmark files.

Supports classic Solomon formats (e.g., C101, R101, RC101) with 25, 50, or 100 customer instances.
Applies mathematical scaling (default factor: 100) on all spatial and temporal dimensions
to satisfy pyvroom's strict integer arithmetic requirements.
"""

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils import DEFAULT_SCALE_FACTOR, compute_distance_matrix, scale_value


@dataclass
class NodeData:
    """Represents a single node (depot or customer) in the VRPTW problem."""
    cust_no: int
    x: float
    y: float
    scaled_x: int
    scaled_y: int
    demand: int
    ready_time: float
    due_date: float
    scaled_ready_time: int
    scaled_due_date: int
    service_time: float
    scaled_service_time: int

    @property
    def coords(self) -> Tuple[float, float]:
        """Original floating-point coordinate pair."""
        return (self.x, self.y)

    @property
    def scaled_coords(self) -> Tuple[int, int]:
        """Integer scaled coordinate pair."""
        return (self.scaled_x, self.scaled_y)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node data to dictionary format."""
        return asdict(self)


@dataclass
class SolomonInstance:
    """Encapsulates a parsed Solomon VRPTW benchmark instance."""
    instance_name: str
    vehicle_count: int
    vehicle_capacity: int
    depot: NodeData
    customers: List[NodeData]
    scale_factor: int = DEFAULT_SCALE_FACTOR
    raw_file_path: Optional[str] = None

    @property
    def all_nodes(self) -> List[NodeData]:
        """Ordered list of all nodes: Depot (index 0) followed by Customers (indices 1..N)."""
        return [self.depot] + self.customers

    @property
    def num_customers(self) -> int:
        """Total number of customer nodes parsed."""
        return len(self.customers)

    @property
    def coordinates(self) -> List[Tuple[float, float]]:
        """List of original (x, y) coordinates for all nodes starting with depot."""
        return [node.coords for node in self.all_nodes]

    @property
    def scaled_coordinates(self) -> List[Tuple[int, int]]:
        """List of scaled (x, y) coordinates for all nodes starting with depot."""
        return [node.scaled_coords for node in self.all_nodes]

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute the scaled 2D integer Euclidean distance/duration matrix."""
        return compute_distance_matrix(
            self.scaled_coordinates,
            scale_factor=self.scale_factor,
            already_scaled=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert full problem instance into dictionary representation."""
        return {
            "instance_name": self.instance_name,
            "vehicle_count": self.vehicle_count,
            "vehicle_capacity": self.vehicle_capacity,
            "scale_factor": self.scale_factor,
            "depot": self.depot.to_dict(),
            "customers": [c.to_dict() for c in self.customers],
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "coordinates": self.coordinates,
            "scaled_coordinates": self.scaled_coordinates,
            "num_customers": self.num_customers,
            "raw_file_path": self.raw_file_path,
        }

    def __getitem__(self, item: str) -> Any:
        """Provide dictionary-like key access for downstream interoperability."""
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(f"Key '{item}' not found in SolomonInstance")

    def __contains__(self, item: str) -> bool:
        return item in self.to_dict()


def parse_solomon_instance(
    file_path: Union[str, Path],
    max_customers: Optional[int] = 25,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
) -> SolomonInstance:
    """
    Parse a standard Solomon VRPTW benchmark instance text file.
    
    Extracts:
      - Vehicle Count
      - Vehicle Capacity
      - Depot data (CUST NO. 0)
      - Customer data (CUST NO. 1..N)
      
    Applies integer scaling (factor 100) to coordinates, time windows, and service times.
    
    Args:
        file_path: Path to the Solomon benchmark text file (e.g., 'data/C101.txt').
        max_customers: Maximum number of customer nodes to extract (default: 25).
                       Set to None to parse all customers present in the file.
        scale_factor: Multiplicative factor for integer discretization (default: 100).
        
    Returns:
        SolomonInstance dataclass object with dictionary-like access.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required sections or data fields cannot be parsed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Solomon instance file not found at: {file_path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [line.strip() for line in f.readlines()]

    # Filter out empty lines
    lines = [line for line in raw_lines if line]

    if not lines:
        raise ValueError(f"Instance file '{file_path}' is empty.")

    # 1. Instance Name (first non-empty line)
    instance_name = lines[0].strip()

    # 2. Extract Vehicle information
    vehicle_count: Optional[int] = None
    vehicle_capacity: Optional[int] = None

    # Scan for VEHICLE section
    vehicle_idx = -1
    for idx, line in enumerate(lines):
        if re.match(r"^VEHICLE", line, re.IGNORECASE):
            vehicle_idx = idx
            break

    if vehicle_idx == -1:
        raise ValueError(f"Could not locate 'VEHICLE' section header in {file_path}")

    # The line following VEHICLE is typically the column headers (NUMBER CAPACITY)
    # The line after that contains the integer values
    val_line_idx = vehicle_idx + 1
    # Advance past headers if present
    while val_line_idx < len(lines) and not re.search(r"\d+", lines[val_line_idx]):
        val_line_idx += 1

    if val_line_idx < len(lines):
        v_tokens = re.findall(r"\d+", lines[val_line_idx])
        if len(v_tokens) >= 2:
            vehicle_count = int(v_tokens[0])
            vehicle_capacity = int(v_tokens[1])

    if vehicle_count is None or vehicle_capacity is None:
        raise ValueError(f"Failed to parse vehicle count and capacity in {file_path}")

    # 3. Extract Customer / Depot section
    customer_idx = -1
    for idx, line in enumerate(lines):
        if re.match(r"^CUSTOMER", line, re.IGNORECASE):
            customer_idx = idx
            break

    if customer_idx == -1:
        raise ValueError(f"Could not locate 'CUSTOMER' section header in {file_path}")

    # Advance past column headers until numerical customer rows begin
    row_idx = customer_idx + 1
    while row_idx < len(lines):
        tokens = lines[row_idx].split()
        # A valid customer row begins with an integer customer number
        if tokens and tokens[0].isdigit():
            break
        row_idx += 1

    depot: Optional[NodeData] = None
    customers: List[NodeData] = []

    while row_idx < len(lines):
        tokens = lines[row_idx].split()
        if not tokens or not tokens[0].isdigit():
            row_idx += 1
            continue

        if len(tokens) < 7:
            # Row has incomplete columns; skip or break
            row_idx += 1
            continue

        cust_no = int(tokens[0])
        x_val = float(tokens[1])
        y_val = float(tokens[2])
        demand_val = int(round(float(tokens[3])))
        ready_time_val = float(tokens[4])
        due_date_val = float(tokens[5])
        service_time_val = float(tokens[6])

        # Mathematical scaling by scale_factor (100) before rounding to integer
        scaled_x = scale_value(x_val, scale_factor)
        scaled_y = scale_value(y_val, scale_factor)
        scaled_ready_time = scale_value(ready_time_val, scale_factor)
        scaled_due_date = scale_value(due_date_val, scale_factor)
        scaled_service_time = scale_value(service_time_val, scale_factor)

        node = NodeData(
            cust_no=cust_no,
            x=x_val,
            y=y_val,
            scaled_x=scaled_x,
            scaled_y=scaled_y,
            demand=demand_val,
            ready_time=ready_time_val,
            due_date=due_date_val,
            scaled_ready_time=scaled_ready_time,
            scaled_due_date=scaled_due_date,
            service_time=service_time_val,
            scaled_service_time=scaled_service_time,
        )

        if cust_no == 0:
            depot = node
        else:
            if max_customers is None or len(customers) < max_customers:
                customers.append(node)

        row_idx += 1

    if depot is None:
        raise ValueError(f"Depot node (CUST NO. 0) not found in {file_path}")

    return SolomonInstance(
        instance_name=instance_name,
        vehicle_count=vehicle_count,
        vehicle_capacity=vehicle_capacity,
        depot=depot,
        customers=customers,
        scale_factor=scale_factor,
        raw_file_path=str(path.resolve()),
    )
