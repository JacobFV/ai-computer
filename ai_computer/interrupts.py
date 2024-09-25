from typing import List, Dict, Any, Callable

from ai_computer.processor import Core, Processor


class Interrupts:
    def __init__(self, processor: "Processor"):
        self.processor = processor
        self.memory = processor.memory
        self.interrupt_table: List[Dict[str, Any]] = []

    def add_interrupt(
        self,
        name: str,
        condition: Callable[[], bool],
        goto_address: int,
        priority: int = 0,
    ) -> None:
        interrupt = {
            "name": name,
            "condition": condition,
            "goto_address": goto_address,
            "priority": priority,
        }
        self.interrupt_table.append(interrupt)
        # Sort interrupts by priority (higher number means higher priority)
        self.interrupt_table.sort(key=lambda x: x["priority"], reverse=True)

    def remove_interrupt(self, name: str) -> None:
        self.interrupt_table = [i for i in self.interrupt_table if i["name"] != name]

    def trigger_interrupt(self, name: str) -> None:
        for interrupt in self.interrupt_table:
            if interrupt["name"] == name:
                # Directly set the instruction pointer of all cores
                for core in self.processor.cores:
                    core.instruction_pointer = interrupt["goto_address"]

    def check_interrupts(self, core: "Core") -> bool:
        for interrupt in self.interrupt_table:
            if interrupt["condition"]():
                # Handle interrupt
                core.instruction_pointer = interrupt["goto_address"]
                return True  # Interrupt handled
        return False  # No interrupt
