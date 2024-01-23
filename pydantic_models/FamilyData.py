from typing import Optional, List

from pydantic import BaseModel
from pydantic_models.Person import Person


class FamilyData(BaseModel):
    life_summary: Optional[str]
    family_members: List[Person]
    main_person: Person

    def concatenate_full_names(self) -> str:
        """
        Concatenates the full names of all family members into a single string.

        This method is useful for generating a comma-separated string of all family member names.
        It should be used when you need a simple and readable representation of all family members' names.

        Returns:
            str: A string containing the full names of all family members, separated by ', '.
        """
        return ", ".join(member.full_name for member in self.family_members)

    def summary_with_names(self) -> str:
        """
        Appends the concatenated full names of family members to the life summary, separated by a specific header.

        This method is useful for creating a comprehensive summary that includes both the life summary and
        a list of family member names. It's particularly useful when you need to display both textual and list-based
        information about a family in a formatted manner.

        Returns:
            str: A string containing the life summary followed by "\n FAMILY MEMBER NAMES: " and the full names of all family members.
        """
        names = self.concatenate_full_names()
        return (
            f"{self.life_summary}\n FAMILY MEMBER NAMES: {names}"
            if self.life_summary
            else f"FAMILY MEMBER NAMES: {names}"
        )
