# Repository Change Notification Workflow

This workflow automatically sends email notifications when important activities occur in the repository.

## Workflow file

```text
.github/workflows/notify-all-repository-changes.yml
```

## Purpose

The workflow helps project members keep track of repository changes without checking GitHub manually.

Whenever a supported event occurs, GitHub Actions sends a formatted email containing:

- Repository name
- Event type
- Action performed
- GitHub user who performed the action
- Event-specific details
- Link to the affected content
- Link to the repository
- Link to the GitHub Actions run

## Monitored events

The workflow monitors the following repository activities:

### Push

Runs when commits are pushed to any branch.

The notification includes:

- Branch name
- Number of commits
- Previous and new commit hashes
- Commit messages
- Commit authors
- Links to commits

### Pull requests

Runs when a pull request is:

- Opened
- Reopened
- Updated
- Edited
- Marked as ready for review
- Converted to draft
- Closed

The notification includes the pull request number, title, source branch, target branch, creator, merge status, and description.

### Issues

Runs when an issue is:

- Opened
- Edited
- Closed
- Reopened
- Assigned or unassigned
- Labeled or unlabeled

The notification includes the issue number, title, creator, state, and content.

### Comments

Runs when a comment on an issue or pull request is:

- Created
- Edited
- Deleted

The notification includes the related issue or pull request, comment author, and comment content.

### Pull request reviews

Runs when a review is:

- Submitted
- Edited
- Dismissed

The notification includes the reviewer, review state, and review content.

### Pull request code comments

Runs when a comment directly attached to a changed code line is created, edited, or deleted.

The notification includes the file path, line number, author, and comment content.

### Releases

Runs when a release is created, edited, published, unpublished, deleted, prereleased, or released.

The notification includes the release tag, name, author, draft status, prerelease status, and release notes.

### Branches and tags

Runs when a branch or tag is created or deleted.

### Manual execution

The workflow can also be started manually from the **Actions** tab through `workflow_dispatch`.

## Email recipients

Notifications are currently sent to:

```text
lethanhloi0603@gmail.com
thainguyenvuquang@gmail.com
```

To add or remove recipients, update the `MAIL_TO` variable in the workflow file.
