# Use a small Nginx image as the base
FROM nginx:alpine

# Copy your frontend file into the Nginx default public directory
COPY index.html /usr/share/nginx/html/index.html

# Expose port 80 for web traffic
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]