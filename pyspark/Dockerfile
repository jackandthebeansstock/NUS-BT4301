FROM quay.io/astronomer/astro-runtime:12.7.1

USER root

# Install OpenJDK-17 in every docker container
RUN apt update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Set JAVA_HOME
# if you are macos user, use ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-arm64
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
RUN export JAVA_HOME

USER astro