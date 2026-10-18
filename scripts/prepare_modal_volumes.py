"""Create the two small paper input/results volumes if they do not exist."""
import modal


def main():
    # The audio volume is expected to exist already; never create an empty replacement.
    modal.Volume.from_name("mtg-jamendo").hydrate()
    for name in ("music-genre-data", "music-genre-runs"):
        modal.Volume.from_name(name, create_if_missing=True).hydrate()
        print(f"Ready: {name}")
    print("Existing mtg-jamendo audio volume is available. No GPU allocated.")


if __name__ == "__main__":
    main()
