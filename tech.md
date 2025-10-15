# 🔍 Debugging Workflows Guide

## 🐍 Python/Streamlit Application Debugging

### Phase 1: Initial Error Identification
1. **Read the full error message** - Don't skip the traceback
2. **Identify the error type** - ImportError, AttributeError, KeyError, etc.
3. **Note the line number** - Find exact location in code
4. **Check recent changes** - What was modified last?

### Phase 2: Common Streamlit Issues

#### API Integration Errors
- **Symptom**: API calls failing or timing out
- **Debug Steps**:
  - Verify API key is correctly stored in `st.session_state`
  - Add try-except blocks with detailed error logging
  - Test API endpoints separately with `st.write()` statements
  - Check rate limits and timeout settings
  - Validate response format before processing

#### State Management Issues
- **Symptom**: Variables resetting, data disappearing
- **Debug Steps**:
  - Confirm `st.session_state` initialization in correct order
  - Add state debugging: `st.sidebar.write(st.session_state)`
  - Check for unintended `st.rerun()` calls
  - Verify cache decorators on functions
  - Look for variable name conflicts

#### File Upload Problems
- **Symptom**: Files not loading, parsing errors
- **Debug Steps**:
  - Print file type: `st.write(uploaded_file.type)`
  - Check file size limits
  - Validate file content before parsing
  - Use `io.StringIO()` or `io.BytesIO()` correctly
  - Test with sample files first

### Phase 3: Systematic Debugging Process

```
1. Isolate the Problem
   ├─ Comment out code sections
   ├─ Add st.write() debug statements
   └─ Test with minimal example

2. Check Dependencies
   ├─ Verify library versions
   ├─ Check import statements
   └─ Review requirements.txt

3. Validate Data Flow
   ├─ Print data at each step
   ├─ Check data types
   └─ Verify transformations

4. Test Edge Cases
   ├─ Empty inputs
   ├─ Large datasets
   └─ Invalid formats
```

## 🛠️ Specific Debug Strategies

### For LLM API Errors
1. Print the full prompt being sent
2. Check token limits
3. Verify model name spelling
4. Test with simpler prompts first
5. Add timeout handling

### For DataFrame Operations
1. Use `st.dataframe()` to inspect data
2. Check column names (spaces, case)
3. Verify data types with `df.dtypes`
4. Test queries on small samples
5. Handle NaN values explicitly

### For UI/Theming Issues
1. Inspect browser console (F12)
2. Test CSS in markdown blocks
3. Check for conflicting styles
4. Verify theme dictionary structure
5. Test on different browsers

## 📋 Debug Checklist

- [ ] Error message copied and read fully
- [ ] Line number identified
- [ ] Recent changes reviewed
- [ ] API keys validated
- [ ] Session state inspected
- [ ] Data types verified
- [ ] Edge cases tested
- [ ] Browser console checked
- [ ] Dependencies confirmed
- [ ] Code commented for testing

## 🚨 Emergency Quick Fixes

**App won't start**: Check imports and API initialization
**Constant reruns**: Look for infinite loops in session state
**Data lost**: Add `st.session_state` persistence
**Slow performance**: Add `@st.cache_data` decorators
**UI broken**: Validate HTML/CSS in markdown blocks

## 💡 Pro Tips

- Use `st.expander()` for debug info in production
- Keep a debug mode toggle in session state
- Log all API calls with timestamps
- Create test data generators for quick testing
- Use `st.spinner()` to show where code is hanging
