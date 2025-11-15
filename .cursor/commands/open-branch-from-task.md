# open-branch

I will give you an Issue ID from linear, and then you will:

1. make sure you are on main branch
2. pull down from the latest main branch
3. create a new worktree with name = Issue ID (i.e. PLA-180) and checkout in the worktree a new branch with format `linear_username/identifiertitle-branchdescription` (i.e. `dov.s/PLA-180-COLMAP-feature-match-preprocessing`). Make sure you stay in the worktree!
4. then start working on the issue
...
5. when you are done, stage and commit your changes, and then using gh cli (already installed) open a PR

*** Do not update the Linear issues status unless I explicitly ask - I have the Github integration setup so the Linear issue should update as a open a branch, commit, make PR, merge PR.
