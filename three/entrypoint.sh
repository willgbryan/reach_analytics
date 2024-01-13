#!/bin/bash

# Check if the OPENAI_API_KEY environment variable is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: The OPENAI_API_KEY environment variable is not set."
  echo "You must set this variable before starting the container."
  echo "Use the following command to set the OPENAI_API_KEY:"
  echo "docker run -e OPENAI_API_KEY='your_api_key' -p 5000:5000 my-python-app"
  exit 1
fi

exec "$@"