import os
import subprocess

BRANCH = "topic-recognition"
REMOTE = "origin"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Show status
print("🔄 Git Status:")
os.system("git status")

# Get commit message
msg = input("\n📝 Commit message: ").strip()
if not msg:
    msg = "Update"

# Add, commit, push
os.system("git add -A")
os.system(f'git commit -m "{msg}"')
os.system(f"git push {REMOTE} {BRANCH}")

print("✅ Done!")
