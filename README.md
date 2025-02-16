# Chatbot-on-AWS-Amplify-with-an-EC2
Deploying an LLM Chatbot on AWS Amplify with an EC2 Backend



Welcome to a comprehensive guide on taking your LLM chatbot from a basic concept to a production-ready application on AWS\! In this tutorial, we will build upon our previous "Hello World" example and architect a chatbot with a Gradio frontend hosted on AWS Amplify and a quantized 7B parameter Large Language Model (LLM) backend running on an Amazon EC2 instance.

**Architecture Overview:**

This production setup will utilize a split architecture:

*   **Frontend (Gradio Chat Interface):**  We'll create a simple HTML/JavaScript frontend, styled with Gradio-like aesthetics, and host it on **AWS Amplify Hosting**. Amplify Hosting is ideal for serving static websites and single-page applications.
*   **Backend (LLM Inference Server):**  The computationally intensive LLM will run on an **Amazon EC2 instance**. EC2 provides the necessary compute power and flexibility to run a 7B parameter quantized model. We will build a simple Flask API on the EC2 instance to serve LLM inferences.

**Why this architecture?**

*   **Scalability and Cost-Effectiveness:** Amplify Hosting is highly scalable and cost-effective for frontends. EC2 provides dedicated resources for the LLM, allowing us to choose instance types based on performance and budget needs.
*   **Separation of Concerns:** Frontend and backend are decoupled, making development, scaling, and maintenance easier.
*   **Production Readiness:** This architecture aligns with common patterns for deploying AI-powered applications in the cloud.

**Important Considerations for Production:**

*   **Resource Requirements:** Running a 7B parameter LLM, even quantized, requires significant resources. Choose an appropriate EC2 instance type (consider GPU instances for better performance).
*   **Security:** Implement proper security measures at all levels (IAM roles, security groups, API authentication, etc.).
*   **Scalability & Reliability:** For true production scale, consider using containerization (Docker, ECS/EKS) for the backend, load balancing, and monitoring. This tutorial provides a foundational setup.
*   **Cost Management:** Monitor AWS resource usage to manage costs effectively, especially for EC2 instances.

## Part 1: Setting up AWS Credentials (IAM User and Amplify CLI Configuration)

We'll reuse the IAM user and Amplify CLI configuration steps from the previous guides, as these are essential for interacting with AWS services.

**Step 1 - Step 5: Create an IAM User and Retrieve Access Keys**

Follow Part 1, Steps 1-5 from the initial "Hello World" guide to create an IAM user (e.g., `prod-llm-chatbot-user`) with `AdministratorAccess` policy (for simplicity in this tutorial, remember to scope down permissions in real production). Securely store your Access Key ID and Secret Access Key.

**Step 6: Configure Amplify CLI**

Run `amplify configure` and follow the prompts from Part 2 of the initial guide, using the credentials of your `prod-llm-chatbot-user` and a profile name like `prod-llm-chatbot-user`.

```bash
amplify configure
```

## Part 2: Setting up the LLM Backend on Amazon EC2

We will launch an EC2 instance and set up a Flask API to serve LLM inferences.

**Step 1: Launch an EC2 Instance**

1.  **Sign in to the AWS Management Console** and navigate to the EC2 service.
2.  Click "Launch instances".
3.  **Choose an AMI (Amazon Machine Image):** Select a suitable AMI. For this tutorial, we'll use Ubuntu Server. Choose a recent Ubuntu Server version.
4.  **Choose an Instance Type:**  This is crucial for LLM performance.
    *   **For best performance (and higher cost):** Consider GPU instances like `g4dn.xlarge`, `g5.xlarge`, or larger GPU instances if needed. These are designed for machine learning workloads.
    *   **For CPU-based (potentially slower, lower cost):**  Choose memory-optimized instances like `r5.large`, `r6i.large`, or similar. Quantization helps, but CPU inference can still be slow for complex models.
    *   For this tutorial, let's assume we choose a **`g4dn.xlarge` (GPU instance) for better demonstration of a 7B model.**  Select this instance type.
5.  **Configure Instance Details:** Accept defaults for most settings. Ensure "Auto-assign Public IP" is **Enabled** so your EC2 instance gets a public IP address for access.
6.  **Add Storage:**  Default storage is usually sufficient for this tutorial.
7.  **Configure Security Group:**  **Crucially, configure the Security Group to allow inbound traffic on port `5000` (for our Flask API) and port `22` (for SSH access).**
    *   Add a rule: Type: "Custom TCP Rule", Port Range: `5000`, Source: "Custom", Source: `0.0.0.0/0` (for open access - **in production, restrict this to your frontend's IP range or use authentication**).
    *   Ensure SSH (port 22) is open for your IP address for instance access.
8.  **Review and Launch:** Review your instance configuration and click "Launch".
9.  **Choose or Create Key Pair:** You'll need a key pair to SSH into your instance. Choose an existing key pair or create a new one and download the `.pem` file. **Store this `.pem` file securely.**
10. Click "Launch Instances".

**Step 2: SSH into your EC2 Instance**

1.  Find the **Public IPv4 address** of your running EC2 instance in the EC2 console.
2.  Open your terminal and use SSH to connect to your instance:

    ```bash
    ssh -i "path/to/your/keypair.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
    ```
    (Replace `"path/to/your/keypair.pem"` with the actual path to your `.pem` file and `<YOUR_EC2_PUBLIC_IP>` with your instance's public IP.)

**Step 3: Install Python and Libraries on EC2**

Once SSHed into your EC2 instance, update the package lists and install Python, pip, and necessary libraries:

```bash
sudo apt update
sudo apt install python3 python3-pip -y
pip3 install flask transformers torch accelerate bitsandbytes gradio  # Install libraries
```

**Step 4: Create the Flask Backend Application (`app.py`) on EC2**

Create a file named `app.py` on your EC2 instance using a text editor like `nano` or `vim`:

```bash
nano app.py
```

Paste the following Python code into `app.py`:

```python
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = Flask(__name__)

# Model and Tokenizer (Adapt for your chosen quantized 7B model - Example with gpt2 for basic demonstration)
# For production, choose a *real* quantized 7B model path
# model_name = "TheBloke/Llama-2-7B-Chat-GGUF" # Example - You would likely use a *specific* GGUF file path/name here if using GGUF directly with a library like llama-cpp-python
# model_basename = "llama-2-7b-chat.Q4_K_S.gguf"

model_name_simple = "gpt2"  # Using gpt2 for basic example - REPLACE with your 7B model
tokenizer = AutoTokenizer.from_pretrained(model_name_simple)
model = AutoModelForCausalLM.from_pretrained(model_name_simple)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1) # device=0 to use GPU if available

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({"error": "Message is required"}), 400

    prompt = "User: " + message + "\n\nAssistant:"
    response_text = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    assistant_response = response_text.split("Assistant:")[-1].strip() # Basic extraction

    return jsonify({"response": assistant_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production
```

**Important adaptations in `app.py` for server deployment:**

*   **Flask Framework:** Imports `Flask`, `request`, `jsonify` to create a REST API.
*   **API Endpoint `/chat`:** Defines a `/chat` endpoint that accepts POST requests.
*   **Request Handling:**  Retrieves the user `message` from the JSON request body.
*   **LLM Inference:**  Uses the same `generator` pipeline as before to get the LLM response.
*   **JSON Response:** Returns the LLM `response` as a JSON object.
*   **`app.run(host='0.0.0.0', port=5000, debug=False)`:** Starts the Flask development server, listening on all interfaces (`0.0.0.0`) on port `5000`.  `debug=False` is crucial for production.

**Step 5: Download the Quantized LLM Model on EC2 (If using a 7B model)**

If you are actually using a 7B quantized model (replace `"gpt2"` in `app.py`), you'll need to ensure the model files are downloaded on your EC2 instance. For models on Hugging Face Hub, `transformers` usually handles downloading when you first run the code. However, for specific quantized formats like GGUF, you might need to manually download the model file to your EC2 instance and adjust the model loading code accordingly.

For this tutorial, we're keeping `"gpt2"` for demonstration purposes as directly running a 7B model on a simple EC2 might be slow and complex to setup instantly.  **Remember to replace `"gpt2"` with your chosen 7B model in `app.py` for a real 7B LLM chatbot.**

**Step 6: Run the Flask Backend on EC2**

In your SSH session on the EC2 instance, run your Flask application:

```bash
python3 app.py
```

You should see Flask start up, indicating your backend server is running on port 5000.  **Keep this terminal session running for the backend to be active.** In a production setup, you would use a process manager like `systemd` or `supervisor` to run the Flask app in the background and ensure it restarts automatically if it crashes.

**Step 7: Test the Backend API (from your local machine)**

Open a new terminal on your **local machine** (not on EC2). You can use `curl` or `Postman` to test the API. Replace `<YOUR_EC2_PUBLIC_IP>` with the public IP of your EC2 instance:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"message": "Hello from curl"}' http://<YOUR_EC2_PUBLIC_IP>:5000/chat
```

You should receive a JSON response back from your EC2 instance containing the LLM's reply. If you get a response, your backend API is working!

## Part 3: Creating the Frontend with AWS Amplify Hosting

Now, let's build a simple frontend and deploy it to Amplify Hosting.

**Step 1: Create Frontend Files ( `index.html`, `script.js`, `style.css` )**

Create a new directory for your frontend project on your **local machine**:

```bash
mkdir llm-chatbot-frontend
cd llm-chatbot-frontend
```

Inside `llm-chatbot-frontend`, create three files: `index.html`, `script.js`, and `style.css`.

**`index.html`:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Simple LLM Chatbot</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="chat-container">
        <div id="chat-log" class="chat-log"></div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Type your message...">
            <button id="send-button">Send</button>
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
```

**`script.js`:**

```javascript
document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    const backendEndpoint = 'http://<YOUR_EC2_PUBLIC_IP>:5000/chat'; // **REPLACE with your EC2 Public IP**

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        appendMessage('user', message);
        userInput.value = '';

        try {
            const response = await fetch(backendEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            appendMessage('bot', data.response);

        } catch (error) {
            appendMessage('error', 'Error communicating with backend: ' + error.message);
            console.error('Error sending message:', error);
        }
    }

    function appendMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        messageDiv.textContent = `${sender === 'user' ? 'You: ' : 'Bot: '}${text}`;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight; // Scroll to bottom
    }
});
```

**Important: Replace `<YOUR_EC2_PUBLIC_IP>` in `script.js` with the actual Public IPv4 address of your EC2 instance.**

**`style.css`:**

```css
.chat-container {
    width: 400px;
    margin: 20px auto;
    border: 1px solid #ccc;
    border-radius: 5px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    height: 500px; /* Fixed height for chat area */
}

.chat-log {
    flex-grow: 1;
    padding: 10px;
    overflow-y: auto; /* Enable vertical scrolling */
    background-color: #f9f9f9;
}

.message {
    padding: 8px;
    margin-bottom: 8px;
    border-radius: 5px;
    clear: both; /* Prevent floating issues */
}

.user {
    background-color: #DCF8C6;
    align-self: flex-end; /* Align user messages to the right */
    float: right;
}

.bot {
    background-color: #ECECEC;
    align-self: flex-start; /* Align bot messages to the left */
    float: left;
}

.error {
    background-color: #FFDDDD; /* Light red for error messages */
    color: darkred;
}


.input-area {
    padding: 10px;
    border-top: 1px solid #ccc;
    display: flex;
}

.input-area input[type="text"] {
    flex-grow: 1;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 5px;
    margin-right: 10px;
}

.input-area button {
    padding: 8px 15px;
    border: none;
    border-radius: 5px;
    background-color: #007bff;
    color: white;
    cursor: pointer;
}

.input-area button:hover {
    background-color: #0056b3;
}
```

**Step 2: Initialize Amplify Project and Add Hosting**

In your `llm-chatbot-frontend` directory, initialize an Amplify project:

```bash
amplify init
```

Follow the `amplify init` prompts (similar to previous guides). Choose "No" for using AWS CloudFormation to create environments as we are just deploying frontend hosting here. Choose "AWS access keys" for authentication and your `prod-llm-chatbot-user` profile.

Then, add Amplify Hosting:

```bash
amplify add hosting
```

Choose "Hosting with Amplify Console", then "Deploy to production", and accept the defaults for other prompts.

**Step 3: Deploy Frontend to Amplify Hosting**

Deploy your frontend application:

```bash
amplify publish
```

Amplify CLI will build your frontend and deploy it to AWS Amplify Hosting. This process will give you an Amplify Hosting URL in the output.

**Step 4: Access Your Production Chatbot**

Open the Amplify Hosting URL provided by `amplify publish` in your web browser. You should now see your chatbot frontend. Type messages, and they will be sent to your EC2 backend, processed by the LLM, and the responses will be displayed in the chat interface!

**Step 5: (Optional) Set up Custom Domain and HTTPS**

For a production website, you'll want to set up a custom domain name and HTTPS. Amplify Hosting makes this relatively straightforward within the Amplify Console. Refer to the AWS Amplify documentation for setting up custom domains and SSL certificates.

**Step 6: (Optional) Clean Up**

To clean up all resources:

1.  **Delete Amplify Hosting Application:** In your `llm-chatbot-frontend` directory, run `amplify delete`.
2.  **Terminate EC2 Instance:** In the EC2 console, terminate your EC2 instance to stop incurring costs.
3.  **Delete IAM User (if desired):** In the IAM console, delete the `prod-llm-chatbot-user` IAM user.

**Production Considerations and Next Steps:**

*   **Security:** Implement API key or more robust authentication for your `/chat` endpoint on EC2. Secure your EC2 instance further by restricting inbound traffic to your frontend's known IP range (if possible) or using a more secure authentication method.
*   **Error Handling and Logging:** Implement more robust error handling in both frontend and backend. Add logging to your backend application for monitoring and debugging.
*   **Scalability and Reliability:** For a production system, consider:
    *   **Backend Scaling:** Use ECS/EKS to containerize your backend application and enable horizontal scaling across multiple instances.
    *   **Load Balancing:** Put a load balancer in front of your backend instances to distribute traffic and improve availability.
    *   **Monitoring and Alerting:** Implement monitoring and alerting for your EC2 instance, backend API, and frontend application.
*   **Database for Conversation History:** For a more feature-rich chatbot, integrate a database (e.g., Amazon DynamoDB) to store conversation history.
*   **Model Optimization and Caching:** Explore further quantization techniques, model optimization, and response caching strategies to improve latency and reduc  ilding a truly production-ready, scalable, and secure application requires further development and attention to best practices in cloud architecture and security. Remember to always prioritize security, monitor your resources, and iterate on your design as you scale your chatbot application. Happy deploying\!
