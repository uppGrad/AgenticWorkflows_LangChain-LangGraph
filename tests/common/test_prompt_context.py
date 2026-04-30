from uppgrad_agentic.common.prompt_context import (
    format_profile_brief,
    format_user_focus,
)


class TestFormatUserFocus:
    def test_empty_input_returns_empty_string(self):
        assert format_user_focus(None) == ""
        assert format_user_focus({}) == ""

    def test_default_intent_is_suppressed(self):
        # "general document feedback" is the no-signal default from
        # parse_user_instructions; on its own it should produce no block.
        assert format_user_focus({"intent": "general document feedback"}) == ""

    def test_renders_intent_tone_role_program(self):
        out = format_user_focus({
            "intent": "tailor for backend role",
            "tone_preferences": ["concise", "technical"],
            "target_role": "Backend Engineer",
            "target_program": "MSc CS at ETH",
            "explicit_constraints": ["keep to one page"],
        })
        assert out.startswith("User focus")
        assert "tailor for backend role" in out
        assert "concise, technical" in out
        assert "Backend Engineer" in out
        assert "MSc CS at ETH" in out
        assert "keep to one page" in out

    def test_filters_parse_fallback_constraint(self):
        out = format_user_focus({
            "intent": "improve clarity",
            "explicit_constraints": ["[parse fallback: LLM failed — boom]"],
        })
        assert "parse fallback" not in out
        # intent still rendered
        assert "improve clarity" in out

    def test_only_default_intent_with_no_other_signal_returns_empty(self):
        out = format_user_focus({
            "intent": "",
            "tone_preferences": [],
            "target_role": None,
            "target_program": None,
            "explicit_constraints": [],
        })
        assert out == ""


class TestFormatProfileBrief:
    def test_empty_input_returns_empty_string(self):
        assert format_profile_brief(None) == ""
        assert format_profile_brief({}) == ""

    def test_renders_target_education_skills_experience(self):
        out = format_profile_brief({
            "target_roles": ["Software Engineer"],
            "education": [{"degree": "BSc CS", "institution": "State U", "year": 2022}],
            "experience": [{"title": "Intern", "company": "TechCorp"}],
            "skills": ["Python", "Docker"],
        })
        assert out.startswith("Applicant profile")
        assert "Software Engineer" in out
        assert "BSc CS, State U, 2022" in out
        assert "Intern @ TechCorp" in out
        assert "Python, Docker" in out

    def test_caps_skills_at_twenty(self):
        many = [f"skill{i}" for i in range(50)]
        out = format_profile_brief({"skills": many})
        assert "skill0" in out
        assert "skill19" in out
        assert "skill20" not in out

    def test_truncates_to_max_chars(self):
        out = format_profile_brief(
            {"skills": ["x" * 100 for _ in range(50)]},
            max_chars=200,
        )
        assert len(out) <= 200
        assert out.endswith("...")

    def test_only_skills_still_renders(self):
        out = format_profile_brief({"skills": ["Python"]})
        assert "Python" in out
        assert out.startswith("Applicant profile")

    def test_renders_extended_fields(self):
        out = format_profile_brief({
            "target_roles": ["Backend Engineer"],
            "experience_level": "ENTRY_LEVEL",
            "location": "Berlin, Germany",
            "education": [
                {"degree": "BSc", "major": "Computer Science",
                 "institution": "State U", "year": 2022},
            ],
            "projects": [{"title": "Distributed log aggregator"}],
            "publications": [{"title": "A paper on consensus"}],
            "achievements": [{"title": "Dean's List"}],
            "languages": ["English", "Spanish"],
            "interests": ["Distributed Systems"],
            "work_style": "REMOTE",
            "work_type": "FULL_TIME",
            "bio": "CS grad focused on tooling.",
        })
        assert "ENTRY_LEVEL" in out
        assert "Berlin, Germany" in out
        assert "in Computer Science" in out  # major rendered separately from degree
        assert "Distributed log aggregator" in out
        assert "A paper on consensus" in out
        assert "Dean's List" in out
        assert "English, Spanish" in out
        assert "Distributed Systems" in out
        assert "REMOTE / FULL_TIME" in out
        assert "CS grad focused on tooling." in out

    def test_education_skips_redundant_major_when_same_as_degree(self):
        # If degree == major (e.g. degree="Computer Science", major="Computer Science")
        # don't render "Computer Science, in Computer Science".
        out = format_profile_brief({
            "education": [
                {"degree": "Computer Science", "major": "Computer Science",
                 "institution": "State U", "year": 2022},
            ],
        })
        assert out.count("Computer Science") == 1

    def test_truncation_drops_bio_first(self):
        # Bio is rendered last; under tight budget, target_roles should survive
        # while bio is the section that gets cut.
        long_bio = "x" * 2000
        out = format_profile_brief(
            {"target_roles": ["Backend Engineer"], "bio": long_bio},
            max_chars=200,
        )
        assert "Backend Engineer" in out
        assert out.endswith("...")
