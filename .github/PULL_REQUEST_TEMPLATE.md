# Pull Request

## Description

Brief description of the changes in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test improvements

## Changes Made

### Python Changes
- [ ] Model conversion improvements
- [ ] Quantization enhancements
- [ ] Performance optimizations
- [ ] Bug fixes
- [ ] New utilities

### Swift Changes
- [ ] iOS API improvements
- [ ] Performance optimizations
- [ ] Bug fixes
- [ ] New features

### Documentation
- [ ] README updates
- [ ] API documentation
- [ ] Code examples
- [ ] Architecture documentation

### Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated  
- [ ] Performance tests added/updated
- [ ] Manual testing completed

## Testing Checklist

### Python Testing
- [ ] All existing tests pass (`make test`)
- [ ] New tests added for new functionality
- [ ] Code coverage maintained (>85%)
- [ ] Linting passes (`make lint`)

### Swift Testing
- [ ] All Swift tests pass (`swift test`)
- [ ] iOS simulator testing completed
- [ ] Memory usage validated
- [ ] Performance benchmarks run

### Integration Testing
- [ ] End-to-end conversion pipeline tested
- [ ] Model accuracy validated
- [ ] Cross-platform compatibility verified

## Performance Impact

### Model Conversion
- [ ] Conversion time: ⬆️ Faster / ➡️ No change / ⬇️ Slower
- [ ] Memory usage: ⬆️ Lower / ➡️ No change / ⬇️ Higher
- [ ] Model size: ⬆️ Smaller / ➡️ No change / ⬇️ Larger

### Inference Performance
- [ ] Latency: ⬆️ Faster / ➡️ No change / ⬇️ Slower
- [ ] Memory usage: ⬆️ Lower / ➡️ No change / ⬇️ Higher
- [ ] Energy usage: ⬆️ Lower / ➡️ No change / ⬇️ Higher

**Performance Details:**
<!-- Provide specific numbers if available -->

## Breaking Changes

- [ ] This PR contains breaking changes

**Breaking Change Details:**
<!-- Describe any breaking changes and migration path -->

## Documentation

- [ ] Code is self-documenting with clear function/class names
- [ ] Public APIs have docstrings
- [ ] README updated if needed
- [ ] API documentation updated if needed
- [ ] Examples updated if needed

## Related Issues

Fixes #(issue_number)
Relates to #(issue_number)

## Screenshots/Examples

<!-- If applicable, add screenshots or code examples -->

## Additional Notes

<!-- Any additional information that reviewers should know -->

## Reviewer Checklist

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Error handling is appropriate
- [ ] Security considerations addressed
- [ ] No hardcoded values or secrets

### Architecture
- [ ] Changes align with project architecture
- [ ] No unnecessary complexity introduced
- [ ] Proper separation of concerns
- [ ] Reusable components where appropriate

### Testing
- [ ] Adequate test coverage
- [ ] Tests are meaningful and maintainable
- [ ] Edge cases considered
- [ ] Performance tests included if applicable

### Documentation
- [ ] Public APIs documented
- [ ] Complex logic explained
- [ ] Examples provided where helpful
- [ ] Changelog updated if needed

---

**For Maintainers:**

### Pre-merge Checklist
- [ ] All CI checks passing
- [ ] Required reviews completed
- [ ] Performance impact acceptable
- [ ] Documentation complete
- [ ] Breaking changes properly communicated

### Post-merge Actions
- [ ] Update project roadmap if applicable  
- [ ] Notify community of significant changes
- [ ] Update deployment documentation if needed
- [ ] Monitor for issues after deployment