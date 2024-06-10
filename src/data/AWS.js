export const AWS = [
// S3
{
title: "What is Amazon S3 and how is it used?",
type: 1,
content: Amazon S3 (Simple Storage Service) is a cloud-based storage service designed for storing and retrieving large amounts of data. It offers a scalable, reliable, and secure solution for storing objects such as files, videos, images, and backups. S3 is used to store and retrieve data from web applications, mobile apps, and big data analytics. It is also commonly used as a data lake to store structured and unstructured data for processing and analysis.
},
{
title: "What is S3 bucket?",
type: 1,
content: An S3 bucket is a container for storing objects in Amazon S3. Buckets are used to organize and manage files and folders in S3. Each bucket must have a globally unique name and is created in a specific region. Access to the objects in a bucket can be controlled using bucket policies, access control lists (ACLs), or both.
},
{
title: "How can you secure your S3 bucket?",
type: 1,
content: There are several ways to secure an S3 bucket, such as:\n - Set up bucket policies to control access to the bucket and its contents - Use AWS Identity and Access Management (IAM) to manage users and permissions - Use server-side encryption to encrypt objects stored in the bucket - Enable versioning to store multiple versions of an object and protect against accidental deletion or overwrite - Configure lifecycle policies to automatically transition objects to lower-cost storage classes or delete them when they are no longer needed.
},
  // Lambda
{
    title: "What is AWS Lambda?",
    type: 1,
    content: `AWS Lambda is a serverless computing service that allows you to run code without provisioning or managing servers. It enables you to execute your code in response to events such as changes to data in Amazon S3, Amazon DynamoDB, or other AWS services, or in response to HTTP requests using API Gateway. Lambda automatically scales the computing resources based on the incoming traffic, allowing you to pay only for the compute time you consume. Lambda supports several programming languages including Node.js, Python, Java, C#, and Go. }, { title: "How is AWS Lambda different from EC2?", type: 1, content:AWS Lambda and Amazon EC2 are both compute services offered by AWS, but they differ in several ways:\n
- AWS Lambda is serverless, meaning you don't need to manage servers, while EC2 requires you to provision, manage, and scale your own servers
- Lambda automatically scales based on the incoming traffic, whereas EC2 requires manual scaling
- Lambda supports event-driven computing, while EC2 provides a more traditional server-based computing model
- Lambda is charged based on the number of requests and duration of execution, while EC2 is charged based on the type and number of instances used. }, { title: "What is the maximum execution time of an AWS Lambda function?", type: 1, content:The maximum execution time of an AWS Lambda function is 15 minutes. If your function requires longer execution time, you should consider breaking it down into smaller, more manageable functions or consider using another compute service such as EC2 or Elastic Beanstalk.`
},
  
// API Gateway
{
    title: "What is Amazon API Gateway?",
    type: 1,
    content: `Amazon API Gateway is a fully managed service that makes it easy to create, publish, maintain, monitor, and secure APIs at any scale. It acts as a "front door" for your application backend, allowing you to create RESTful APIs and WebSocket APIs that enable real-time communication between your application and connected clients. API Gateway supports various integrations with AWS services, such as Lambda, DynamoDB, and S3, as well as with third-party services.`
},
{
    title: "What are the benefits of using Amazon API Gateway?",
    type: 1,
    content: `The benefits of using Amazon API Gateway include:\n
        - Easy creation of RESTful APIs and WebSocket APIs
        - Seamless integration with AWS services and third-party services
        - Scalability and high availability through automatic scaling and caching
        - Security and identity management through integration with AWS Identity and Access Management (IAM) and OAuth 2.0
        - Detailed monitoring and logging of API usage and performance.`
},
{
    title: "How can you secure your API Gateway?",
    type: 1,
    content: `You can secure your API Gateway using various methods, such as:\n
        - Configuring authorization and authentication mechanisms, such as IAM roles and policies, Lambda authorizers, and OAuth 2.0
        - Using HTTPS for secure communication between API Gateway and your clients
        - Enabling throttling and rate limiting to protect against DoS attacks and to prevent abuse
        - Configuring logging and monitoring to track API usage and detect anomalies
        - Implementing encryption at rest and in transit to protect sensitive data.`
},

// CloudFront
{
    title: "What is Amazon CloudFront?",
    type: 1,
    content: `Amazon CloudFront is a content delivery network (CDN) that distributes static and dynamic content, such as HTML, CSS, JavaScript, images, and videos, to users worldwide. It improves the performance and scalability of your application by caching content at edge locations located closer to the users, reducing latency and network traffic. CloudFront integrates with other AWS services, such as S3, EC2, and Lambda, and supports custom SSL certificates and access control through AWS Identity and Access Management (IAM).`
},
{
    title: "What are the benefits of using Amazon CloudFront?",
    type: 1,
    content: `The benefits of using Amazon CloudFront include:\n
        - Faster content delivery through caching and edge locations
        - Improved performance and scalability of your application
        - Reduced latency and network traffic
        - Seamless integration with other AWS services and custom SSL certificates
        - Security and access control through AWS Identity and Access Management (IAM)
        - Detailed monitoring and logging of CDN usage and performance.`
},
{
    title: "What is the difference between Amazon CloudFront and Amazon S3?",
    type: 1,
    content: `Amazon CloudFront and Amazon S3 are both storage and content delivery services offered by AWS, but they differ in several ways:\n
        - Amazon S3 is a storage service for storing and retrieving objects, while CloudFront is a CDN for distributing content
        - S3 provides a simple web services interface that can be used to store and retrieve any amount of data, while CloudFront is designed for high-speed content delivery with edge caching and dynamic content routing
        - S3 is typically used for storing and serving static content, while CloudFront is used for dynamic content and content that is frequently accessed by users
- S3 supports multiple storage classes for different access patterns and cost optimization, while CloudFront supports edge caching and origin shielding for improved performance and security
- S3 can be accessed directly by clients, while CloudFront serves as a global content distribution network that caches content at edge locations closer to users
- CloudFront integrates with other AWS services, such as S3, EC2, and Lambda, while S3 can be used independently or in conjunction with other AWS services.`
},
  // CloudWatch
{
    title: "What is Amazon CloudWatch?",
    type: 1,
    content: `Amazon CloudWatch is a monitoring and observability service that provides metrics, logs, and alarms for AWS resources and applications. It collects and tracks performance data, such as CPU usage, network traffic, and disk I/O, and provides real-time visibility into the health and performance of your infrastructure. CloudWatch supports various types of data sources, including EC2 instances, Lambda functions, RDS databases, and custom metrics, and integrates with other AWS services, such as SNS and Auto Scaling, for automated actions and notifications.`
},
{
    title: "What are the benefits of using Amazon CloudWatch?",
    type: 1,
    content: `The benefits of using Amazon CloudWatch include:\n
        - Real-time monitoring and visibility into AWS resources and applications
        - Collection and tracking of performance data and metrics
        - Customizable alarms and notifications for automated actions
        - Integration with other AWS services and third-party tools
        - Easy creation of dashboards and visualizations for data analysis
        - Scalability and high availability through automatic scaling and regional replication.`
},
{
    title: "What types of data can you monitor with Amazon CloudWatch?",
    type: 1,
    content: `You can monitor various types of data with Amazon CloudWatch, including:\n
        - Metrics: numerical data that represent the performance of a resource or application, such as CPU utilization, network traffic, and request latency
        - Logs: text-based data that contain information about events, errors, and transactions that occur within a resource or application, such as Apache access logs, AWS CloudTrail logs, and custom application logs
        - Events: notifications and alerts triggered by changes in a resource or application, such as EC2 instance termination or S3 bucket creation
        - Alarms: customized notifications and actions triggered by threshold breaches or anomalies in metric data, such as sending an email or invoking a Lambda function.`
},

// SNS/SQS/SES
{
    title: "What is Amazon SNS?",
    type: 1,
    content: `Amazon SNS (Simple Notification Service) is a fully managed messaging service that enables the pub/sub (publish/subscribe) messaging pattern for distributed applications and microservices. It allows you to send and receive messages between AWS services, third-party services, and mobile devices in a highly scalable and reliable manner. SNS supports various protocols, such as HTTP, HTTPS, email, SMS, and mobile push notifications, and integrates with other AWS services, such as Lambda, CloudFormation, and CloudWatch.`
},
{
    title: "What is Amazon SQS?",
    type: 1,
    content: `Amazon SQS (Simple Queue Service) is a fully managed message queuing service that enables the decoupling of components and services within a distributed system. It allows you to send, receive, and process messages between software components, microservices, and AWS services in a highly scalable and reliable manner. SQS supports both standard and FIFO (first-in, first-out) queues, and integrates with other AWS services, such as Lambda, EC2, and S3, for triggering and processing messages. }, { title: "What is Amazon SES?", type: 1, content:Amazon SES (Simple Email Service) is a scalable and cost-effective email service that enables you to send and receive email messages using your own email addresses or domains. It provides a reliable and secure email infrastructure that integrates with other AWS services, such as SNS, Lambda, and CloudWatch, and supports various email protocols, such as SMTP, SMTPS, and Amazon SES API. SES also includes features for email authentication, bounce and complaint handling, and message tracking and reporting. }, { title: "What are the differences between Amazon SNS and Amazon SQS?", type: 1, content:Amazon SNS and Amazon SQS are both messaging services that enable distributed applications and microservices to communicate with each other, but they have different characteristics and use cases. The main differences between SNS and SQS are:\n
- SNS uses the pub/sub (publish/subscribe) messaging pattern, while SQS uses the queue-based messaging pattern
- SNS delivers messages to multiple recipients simultaneously, while SQS delivers messages to one consumer at a time
- SNS supports various protocols, such as HTTP, HTTPS, email, SMS, and mobile push notifications, while SQS supports only the Amazon SQS API
- SNS is suitable for real-time notifications and event-driven architectures, while SQS is suitable for asynchronous processing and decoupling of components and services.`
},
  // CLI
{
    title: "What is the AWS CLI?",
    type: 1,
    content: `The AWS CLI (Command Line Interface) is an open-source tool that enables you to interact with AWS services and resources from the command line or from scripts. It provides a unified interface for managing AWS resources, such as EC2 instances, S3 buckets, and IAM users, and supports various commands for performing operations, such as creating, deleting, updating, and querying resources. The AWS CLI also includes features for authentication and authorization, profile management, output formatting, and integration with other CLI tools, such as bash and PowerShell.`
},
{
    title: "What are the benefits of using the AWS CLI?",
    type: 1,
    content: `The benefits of using the AWS CLI include:\n
        - Automation and scripting of AWS operations and tasks
        - Consistency and repeatability of AWS configurations and deployments
        - Efficient management of AWS resources and services
        - Integration with other CLI tools and scripts for advanced workflows and data processing
        - Support for multiple platforms and operating systems
        - Open-source community contributions and extensions.`
},
{
    title: "What are some common AWS CLI commands?",
    type: 1,
    content: `Some common AWS CLI commands include:\n
        - aws configure: configure the AWS CLI with your access keys, region, and output format
        - aws s3: manage S3 buckets and objects, such as creating, deleting, copying, and listing files
        - aws ec2: manage EC2 instances, such as launching, stopping, and terminating instances
        - aws lambda: manage Lambda functions, such as creating, updating, and invoking functions
        - aws cloudformation: manage CloudFormation stacks, such as creating, updating, and deleting stacks
        - aws cloudwatch: manage CloudWatch metrics, alarms, and logs, such as creating, querying, and analyzing logs.`
},

// General
{
    title: "What is the AWS Well-Architected Framework?",
type: 1,
content: The AWS Well-Architected Framework is a set of best practices, guidelines, and recommendations for designing, building, and operating reliable, secure, efficient, and cost-effective AWS workloads. The framework provides a systematic approach for evaluating and improving the architecture of AWS applications and services based on five pillars:\n - Operational excellence: the ability to run and monitor systems to deliver business value and to continually improve processes and procedures - Security: the ability to protect information, systems, and assets while delivering business value through risk assessments and mitigation strategies - Reliability: the ability to recover from failures and to meet business and customer demand through fault-tolerant and scalable architectures and processes - Performance efficiency: the ability to use computing resources efficiently and to maintain the required performance level through optimization and scalability strategies - Cost optimization: the ability to avoid unnecessary costs and to use resources in the most effective way through cost-aware architectures and processes.
},
{
title: "What is AWS CloudFormation?",
type: 1,
content: AWS CloudFormation is a service that enables you to automate the creation and management of AWS resources and applications using templates. CloudFormation templates are JSON or YAML files that define the infrastructure, services, and dependencies of an application or workload. With CloudFormation, you can create, update, and delete resources and stacks in a consistent and repeatable way, and you can also version control and share templates with other users and teams. CloudFormation integrates with other AWS services, such as EC2, S3, Lambda, and IAM, and provides features for parameterization, rollbacks, nested stacks, and drift detection.
},
{
title: "What is AWS CloudTrail?",
type: 1,
content: AWS CloudTrail is a service that enables you to log, monitor, and audit the activity of AWS resources and services in your AWS account. CloudTrail captures API calls and events from various AWS services and delivers them to a S3 bucket or a CloudWatch Logs group for storage and analysis. CloudTrail provides a detailed record of actions taken by users, roles, and services in your account, including resource changes, security events, and compliance information. CloudTrail can also be used for troubleshooting, security analysis, and compliance audits, and integrates with other AWS services, such as CloudWatch, SNS, and Lambda.
}
];
