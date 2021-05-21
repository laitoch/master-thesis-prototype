from typing import Optional


def query_user_yes_no(default: Optional[bool] = None) -> bool:
    """
    Get a yes or no answer from the user.
    """
    yes = {"yes", "y", "ye"}
    no = {"no", "n"}

    while True:
        choice = input().lower()
        if choice == "" and default is not None:
            return default
        elif choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond with 'yes/y' or 'no/n'")
