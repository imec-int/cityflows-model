#!/bin/bash

echo "Verifying if tag ($GITVERSION_SEMVER) should be pushed to git for branch: $BUILD_SOURCEBRANCHNAME"

if [ "$BUILD_SOURCEBRANCHNAME" == "master" ]; then
	echo "Tagging on git..."
	VERSION="$GITVERSION_SEMVER"
	git config --global user.email "apt.devops@imec.be"
	git config --global user.name "AzureDevops"
	git push --delete origin $VERSION
	git tag -d $VERSION
	git tag -a $VERSION -m "Release $GITVERSION_SEMVER"
	git push origin $VERSION
fi
