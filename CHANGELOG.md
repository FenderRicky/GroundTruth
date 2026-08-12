# Changelog

All notable changes to GroundTruth are documented here.

## [2026-08-12]

### Improved

* Documented the current analysis workflow and project behavior.
* Clarified that profile analysis combines real GitHub activity with uploaded resume data.
* Added a changelog to make future project updates easier to track.

### Project Notes

* GitHub activity is retrieved through the GitHub REST API.
* Resume content is extracted from uploaded PDF files.
* Optional target-job information can be used to improve the analysis context.
* Analysis is generated through the configured LLM provider.
* The application is designed to run as a lightweight FastAPI/Vercel deployment.

---
