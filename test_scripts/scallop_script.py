from scallopy import ScallopContext


def main():
    ctx = ScallopContext()

    # 1) Base kinship facts (now including Bob)
    ctx.add_relation("kinship", (str, str, str))
    ctx.add_facts("kinship", [
        ("Alice", "mother", "Beth"),
        ("Bob", "father", "Beth"),
        ("Beth", "mother", "Charlie"),
        ("Charlie", "father", "David"),
    ])

    # 2) Derive a generic 'parent' relation from both mother/father
    ctx.add_relation("parent", (str, str))
    ctx.add_rule("parent(A, C) = kinship(A, \"mother\", C)")
    ctx.add_rule("parent(A, C) = kinship(A, \"father\", C)")

    # 3) Composition table for grand-relations
    ctx.add_relation("composed", (str, str, str))
    ctx.add_facts("composed", [
        ("mother", "parent", "grandmother"),
        ("father", "parent", "grandfather"),
    ])

    # 4) Infer all grand-kin via one chaining rule
    #    kin(A, R3, C) :- kinship(A, R1, B), parent(B, C), composed(R1, 'parent', R3)
    ctx.add_rule(
        "kin(A, R3, C) = "
        "kinship(A, R1, B), "
        "parent(B, C), "
        "composed(R1, \"parent\", R3)"
    )

    # 5) Run and print
    ctx.run()
    print("Inferred kin:")
    for subj, rel, obj in ctx.relation("kin"):
        print(f"  {subj} --{rel}--> {obj}")

if __name__ == "__main__":
    main()

