//! Hugging Face Hub repository identifier validation.
//!
//! Mirrors the rules enforced by `huggingface_hub`'s `validate_repo_id`
//! (`utils/_validators.py`) so that handle strings constructed via
//! [`HFClient::model`], [`HFClient::dataset`], etc., can be screened against
//! the same shape rules the Hub itself applies.
//!
//! The validator operates on a single segment (owner or name) at a time
//! because that is what callers typically have in hand — the two-segment
//! `owner/name` form is split before storage in handle types.
//!
//! [`HFClient::model`]: crate::HFClient::model
//! [`HFClient::dataset`]: crate::HFClient::dataset

use std::fmt;

/// Which side of an `owner/name` repository identifier a segment came from.
///
/// Used in [`RepoIdValidationError`] messages and as the second argument to
/// [`validate_repo_id_segment`]. Affects only the `.git` suffix rule, which
/// applies to `name` but not `owner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentRole {
    /// The namespace portion (user or organization).
    Owner,
    /// The repository name portion.
    Name,
}

impl fmt::Display for SegmentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SegmentRole::Owner => f.write_str("owner"),
            SegmentRole::Name => f.write_str("name"),
        }
    }
}

/// Why a repository ID segment failed [`validate_repo_id_segment`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RepoIdValidationError {
    /// The segment was the empty string.
    #[error("the {role} segment must not be empty")]
    Empty { role: SegmentRole },
    /// The segment exceeded the 96-character limit.
    #[error("the {role} segment must be at most 96 characters (got {length})")]
    TooLong { role: SegmentRole, length: usize },
    /// The segment contained a character outside `[A-Za-z0-9._-]`.
    #[error("the {role} segment contains an invalid character: '{character}'")]
    InvalidCharacter { role: SegmentRole, character: char },
    /// The segment started or ended with `.`.
    #[error("the {role} segment must not start or end with '.'")]
    LeadingOrTrailingDot { role: SegmentRole },
    /// The segment started or ended with `-`.
    #[error("the {role} segment must not start or end with '-'")]
    LeadingOrTrailingHyphen { role: SegmentRole },
    /// The segment contained `--`.
    #[error("the {role} segment must not contain '--'")]
    DoubleHyphen { role: SegmentRole },
    /// The segment contained `..`.
    #[error("the {role} segment must not contain '..'")]
    DoubleDot { role: SegmentRole },
    /// The `name` segment ended with `.git`.
    #[error("the name segment must not end with '.git'")]
    GitSuffix,
}

/// Validate a single owner or name segment against the Hugging Face Hub naming
/// rules.
///
/// Returns `Ok(())` if `segment` is well-formed for `role`, otherwise the
/// specific [`RepoIdValidationError`] that fired first. Validation is in a
/// fixed order: emptiness → length → character set → boundary chars → no
/// `--`/`..` → `.git` suffix (name only).
///
/// Rules:
///
/// - 1–96 characters long.
/// - Characters drawn from `[A-Za-z0-9._-]`.
/// - Must not start or end with `.` or `-`.
/// - Must not contain `--` or `..`.
/// - For `SegmentRole::Name`: must not end with `.git`.
pub fn validate_repo_id_segment(segment: &str, role: SegmentRole) -> Result<(), RepoIdValidationError> {
    if segment.is_empty() {
        return Err(RepoIdValidationError::Empty { role });
    }
    let length = segment.chars().count();
    if length > 96 {
        return Err(RepoIdValidationError::TooLong { role, length });
    }
    for character in segment.chars() {
        if !is_allowed_character(character) {
            return Err(RepoIdValidationError::InvalidCharacter { role, character });
        }
    }
    if segment.starts_with('.') || segment.ends_with('.') {
        return Err(RepoIdValidationError::LeadingOrTrailingDot { role });
    }
    if segment.starts_with('-') || segment.ends_with('-') {
        return Err(RepoIdValidationError::LeadingOrTrailingHyphen { role });
    }
    if segment.contains("--") {
        return Err(RepoIdValidationError::DoubleHyphen { role });
    }
    if segment.contains("..") {
        return Err(RepoIdValidationError::DoubleDot { role });
    }
    if role == SegmentRole::Name && segment.ends_with(".git") {
        return Err(RepoIdValidationError::GitSuffix);
    }
    Ok(())
}

fn is_allowed_character(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_'
}

#[cfg(test)]
mod tests {
    use super::{RepoIdValidationError, SegmentRole, validate_repo_id_segment};

    #[test]
    fn accepts_typical_owner_and_name() {
        assert!(validate_repo_id_segment("openai-community", SegmentRole::Owner).is_ok());
        assert!(validate_repo_id_segment("gpt2", SegmentRole::Name).is_ok());
        assert!(validate_repo_id_segment("HuggingFaceFW", SegmentRole::Owner).is_ok());
        assert!(validate_repo_id_segment("fineweb-edu", SegmentRole::Name).is_ok());
    }

    #[test]
    fn rejects_empty_segment() {
        assert!(matches!(
            validate_repo_id_segment("", SegmentRole::Owner),
            Err(RepoIdValidationError::Empty {
                role: SegmentRole::Owner
            })
        ));
    }

    #[test]
    fn rejects_segment_over_96_characters() {
        let long = "a".repeat(97);
        assert!(matches!(
            validate_repo_id_segment(&long, SegmentRole::Name),
            Err(RepoIdValidationError::TooLong {
                length: 97,
                role: SegmentRole::Name
            })
        ));
    }

    #[test]
    fn accepts_segment_at_96_character_boundary() {
        let exactly_96 = "a".repeat(96);
        assert!(validate_repo_id_segment(&exactly_96, SegmentRole::Name).is_ok());
    }

    #[test]
    fn rejects_invalid_character() {
        let err = validate_repo_id_segment("foo/bar", SegmentRole::Name).unwrap_err();
        assert!(matches!(
            err,
            RepoIdValidationError::InvalidCharacter {
                role: SegmentRole::Name,
                character: '/'
            }
        ));
        // Spaces, slashes, and other punctuation are all rejected.
        assert!(validate_repo_id_segment("foo bar", SegmentRole::Name).is_err());
        assert!(validate_repo_id_segment("foo@bar", SegmentRole::Name).is_err());
        // Non-ASCII letters are not allowed (the Hub uses ASCII identifiers).
        assert!(validate_repo_id_segment("fooé", SegmentRole::Name).is_err());
    }

    #[test]
    fn rejects_leading_or_trailing_dot() {
        assert!(matches!(
            validate_repo_id_segment(".foo", SegmentRole::Name),
            Err(RepoIdValidationError::LeadingOrTrailingDot { .. })
        ));
        assert!(matches!(
            validate_repo_id_segment("foo.", SegmentRole::Name),
            Err(RepoIdValidationError::LeadingOrTrailingDot { .. })
        ));
    }

    #[test]
    fn rejects_leading_or_trailing_hyphen() {
        assert!(matches!(
            validate_repo_id_segment("-foo", SegmentRole::Name),
            Err(RepoIdValidationError::LeadingOrTrailingHyphen { .. })
        ));
        assert!(matches!(
            validate_repo_id_segment("foo-", SegmentRole::Name),
            Err(RepoIdValidationError::LeadingOrTrailingHyphen { .. })
        ));
    }

    #[test]
    fn rejects_double_hyphen() {
        assert!(matches!(
            validate_repo_id_segment("foo--bar", SegmentRole::Name),
            Err(RepoIdValidationError::DoubleHyphen { .. })
        ));
    }

    #[test]
    fn rejects_double_dot() {
        assert!(matches!(
            validate_repo_id_segment("foo..bar", SegmentRole::Name),
            Err(RepoIdValidationError::DoubleDot { .. })
        ));
    }

    #[test]
    fn rejects_git_suffix_on_name_only() {
        assert!(matches!(
            validate_repo_id_segment("model.git", SegmentRole::Name),
            Err(RepoIdValidationError::GitSuffix)
        ));
        // The same string passes for an owner — only `name` rejects `.git`.
        // (Owners can theoretically end in `.git` per the Hub's identifier
        // rules; in practice they rarely do, but the suffix check is
        // name-scoped.)
        // Note: this still trips the leading-or-trailing-dot rule if the
        // segment is just ".git" so we use a non-trivial owner here.
        assert!(validate_repo_id_segment("model.git", SegmentRole::Owner).is_ok());
    }

    #[test]
    fn error_messages_contain_role() {
        let err = validate_repo_id_segment("", SegmentRole::Owner).unwrap_err();
        assert_eq!(err.to_string(), "the owner segment must not be empty");

        let err = validate_repo_id_segment(".foo", SegmentRole::Name).unwrap_err();
        assert_eq!(err.to_string(), "the name segment must not start or end with '.'");
    }
}
