#!/bin/bash
echo "UPDATING environment.lock.yml"

cat > .git/hooks/pre-commit << EOF
# .git/hooks/pre-commit
#!/bin/bash
if git diff --cached --name-only | grep -q "environment.yml"; then
    echo "environment.yml changed — updating lock file..."
    conda env export --no-builds | grep -v "^prefix:" > environment.lock.yml
    git add environment.lock.yml
fi
EOF
chmod +x ./.git/hooks/pre-commit