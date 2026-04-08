# server/graders.py

def grade_style_review(state) -> float:
    ground_truth = state.ground_truth_issues
    comments = state.comments
    files_read = state.files_read

    total_issues = len(ground_truth)
    if total_issues == 0:
        return 0.0

    # Build a set of (file, line) tuples from ground truth
    gt_set = {(issue["file"], issue["line"]) for issue in ground_truth}

    correct = 0
    false_positives = 0
    for comment in comments:
        key = (comment["file"], comment["line"])
        if key in gt_set:
            correct += 1
        else:
            false_positives += 1

    # Coverage bonus: all unique files from ground truth must appear in files_read
    gt_files = {issue["file"] for issue in ground_truth}
    files_read_set = set(files_read)
    coverage_bonus = 1.0 if gt_files.issubset(files_read_set) else 0.0

    score = (correct / total_issues) * 0.7 + coverage_bonus * 0.3 - 0.05 * false_positives
    return float(max(0.0, min(1.0, score)))


def grade_logic_bugs(state) -> float:
    ground_truth = state.ground_truth_issues
    comments = state.comments
    info = state.info if hasattr(state, "info") else {}

    bug_weights = {
        "off_by_one": 0.25,
        "race_condition": 0.40,
        "edge_case": 0.35,
    }

    fix_keywords = {
        "off_by_one": [">="],
        "race_condition": ["await"],
        "edge_case": ["null"],
    }

    # Build map from (file, line) -> bug type
    gt_map = {(issue["file"], issue["line"]): issue["type"] for issue in ground_truth}

    score = 0.0
    semantic_bonus_granted = False

    for comment in comments:
        key = (comment["file"], comment["line"])
        if key in gt_map:
            bug_type = gt_map[key]
            score += bug_weights.get(bug_type, 0.0)
            # Check for semantic fix keyword
            if not semantic_bonus_granted and bug_type in fix_keywords:
                text = comment.get("text", "")
                for kw in fix_keywords[bug_type]:
                    if kw in text:
                        semantic_bonus_granted = True
                        break

    if semantic_bonus_granted:
        score += 0.10

    assigned_score = info.get("assigned_score", 10)
    if assigned_score <= 4:
        score += 0.10

    return float(max(0.0, min(1.0, score)))


def grade_security_audit(state) -> float:
    ground_truth = state.ground_truth_issues
    comments = state.comments
    info = state.info if hasattr(state, "info") else {}

    vuln_weights = {
        "sql_injection": 0.25,
        "insecure_deserialization": 0.25,
        "path_traversal": 0.20,
        "hardcoded_secret": 0.15,
    }

    severity_keywords = {
        "sql_injection": ["critical"],
        "insecure_deserialization": ["critical"],
        "path_traversal": ["high"],
        "hardcoded_secret": ["medium"],
    }

    # Build map from (file, line) -> vuln type
    gt_map = {(issue["file"], issue["line"]): issue["type"] for issue in ground_truth}

    score = 0.0
    sql_injection_found = False
    correctly_classified_severities = 0

    for comment in comments:
        key = (comment["file"], comment["line"])
        if key in gt_map:
            vuln_type = gt_map[key]
            score += vuln_weights.get(vuln_type, 0.0)

            if vuln_type == "sql_injection":
                sql_injection_found = True

            # Check severity classification
            text = comment.get("text", "").lower()
            for kw in severity_keywords.get(vuln_type, []):
                if kw in text:
                    correctly_classified_severities += 1
                    break

    score += 0.05 * correctly_classified_severities

    assigned_score = info.get("assigned_score", 5)
    if sql_injection_found and assigned_score == 0:
        score += 0.15

    return float(max(0.0, min(1.0, score)))
