import os;
import streamlit as st
import joblib
from PIL import Image
import numpy as np

from custom_module.preprocess_image import preprocess_image
from custom_module.resize_digit import resize_digit
from custom_module.find_digits import find_digits
from custom_module.text_to_speech import text_to_speech

# Get the base directory of the project
base_dir = os.path.dirname(__file__)


# Construct the relative paths to the model files
Nmodel_model_path = os.path.join(base_dir, 'model', 'Nmodel.joblib')


# Load the trained machine learning models
loaded_model = joblib.load(Nmodel_model_path)

# Load the model
#loaded_model = joblib.load('model/Nmodel.joblib')

# Sidebar
st.sidebar.title("Handwritten Number Prediction App")

# Set the initial section to "Home"
selected_section = st.sidebar.radio(" ", ["Home", "Motivation", "Predictor System", "Collaborators", "Recommendations"])
st.sidebar.markdown("---")

# Section: Home
if selected_section == "Home":
    st.title("Home")
    st.image("static/images/home.jpg", use_column_width=True)
    st.write("Welcome to our Handwritten Number Prediction App, a space where cutting-edge technology meets the timeless art of handwritten digits. Our platform leverages the renowned MNIST handwritten digit classification dataset, pushing the boundaries of predictive modeling to bring you accurate and efficient handwritten number predictions.")
    
    st.write("In the realm of machine learning, the MNIST dataset is a cornerstone, often considered a rite of passage for aspiring data scientists and engineers. Our app takes this fundamental dataset and elevates its application, showcasing the versatility and potential of predictive models in deciphering handwritten numbers. The elegance of the MNIST dataset lies in its simplicity – a collection of 28x28 pixel grayscale images of handwritten digits (0-9), serving as the ideal playground for training and testing machine learning algorithms.")
   
    st.write("As you navigate through our app, you'll witness the seamless integration of sophisticated algorithms with user-friendly interfaces. The predictive system at the core of our application not only showcases accurate predictions but also provides a peek into the fascinating world of image processing, feature extraction, and model inference.")
    
    st.write("The heart of our app lies in the Predictor System section. Upload an image containing a handwritten number, and watch as our model works its magic. Behind the scenes, the image is preprocessed, digits are identified, and our machine learning model makes predictions with precision. It's not just about predicting numbers; it's about demystifying the intricate process of transforming raw data into meaningful insights.")
    
    st.write("Our journey doesn't end with accurate predictions. Explore the Recommendations section to discover the potential future collaborations and the exciting possibilities that lie ahead. We envision this project as a dynamic hub for collaboration, where enthusiasts, researchers, and industry professionals can come together to explore, innovate, and redefine the landscape of handwritten digit recognition.")
    
    st.write("Whether you're here to make predictions, explore the intricacies of machine learning, or join us in shaping the future of this project, we welcome you. Explore, engage, and discover the endless possibilities that our Handwritten Number Prediction App unfolds. The journey has just begun, and we invite you to be a part of this exciting venture into the convergence of tradition and technology.")

# Section: Motivation
elif selected_section == "Motivation":
    st.title("Motivation")

    st.write("## Enhancing Handwritten Number Prediction with MNIST")
    st.write("Welcome to our Handwritten Number Prediction App! Our motivation lies in leveraging the powerful MNIST handwritten "
             "digit classification dataset to advance the field of handwritten number prediction. While the MNIST dataset has been "
             "a cornerstone in the development of machine learning algorithms, we aim to take it a step further and showcase the potential "
             "of predictive models for recognizing handwritten numbers.")

    st.write("### Building on a Classic Benchmark")
    st.write("The MNIST dataset, comprising 28x28 pixel grayscale images of handwritten digits, has been a cornerstone in the development "
             "of machine learning algorithms. Our motivation is to build on this classic benchmark and demonstrate how modern techniques can "
             "enhance the accuracy and efficiency of handwritten number prediction.")

    st.write("### Realizing Practical Applications")
    st.write("Beyond a mere academic exercise, our app focuses on the practical applications of handwritten number prediction. From recognizing "
             "handwritten digits on forms to aiding in document digitization, we strive to showcase the real-world utility of predictive models "
             "trained on the MNIST dataset.")

    st.write("### Exploring Advanced Techniques")
    st.write("Our journey goes beyond basic digit recognition. We delve into advanced techniques, exploring sophisticated models and preprocessing "
             "methods to achieve superior prediction results. By doing so, we aim to contribute to the broader landscape of machine learning advancements.")

    st.write("### Bridging the Gap for Users")
    st.write("Whether you're a seasoned data scientist or someone new to machine learning, our app is designed to bridge the gap and make the power "
             "of predictive models accessible. We invite users to explore, learn, and witness the capabilities of MNIST-based models in predicting "
             "handwritten numbers accurately.")


# Section: Predictor System
elif selected_section == "Predictor System":
    st.title("Predictor System")
    uploaded_file = st.file_uploader("Choose an image of a handwritten number", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        preprocessed_img = preprocess_image(np.array(image))
        digit_images = find_digits(preprocessed_img)
        digit_images = np.array(digit_images)
        digit_images = digit_images / 255.0

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        digits_pred = loaded_model.predict(digit_images)
        prediction = "".join([str(np.argmax(i)) for i in digits_pred])

        st.write(f"Prediction: {prediction}")

        # Speech Button
        if st.button("Speak Prediction"):
            # Speak the prediction
            text_to_speech(f"The given number is {prediction}")

# Section: Collaborators
elif selected_section == "Collaborators":
    st.title("Collaborators")
    st.write("Step into the collaborative realm where innovation meets expertise, and witness the collective efforts of the brilliant minds shaping our project. Our team comprises dedicated individuals, each contributing their unique skills and perspectives to create something extraordinary.")
    
    st.write("Our collaboration is a testament to the power of teamwork, innovation, and shared passion. Connect with us, explore our work, and witness the collective spirit that fuels the success of our project.")
    
    collaborators_info = [
        {
            "name": "Tofayel Ahmmed Babu",
            "github": "https://github.com/TofayelAhmmedBabu",
            "linkedin": "https://www.linkedin.com/in/tofayelahmmedbabu/",
            "portfolio": "https://tofayelahmmedbabu.vercel.app/",
        },
        {
            "name": "Md. Refaj Hossan",
            "github": "https://github.com/RJ-Hossan",
            "linkedin": "https://www.linkedin.com/in/mdrefajhossan/",
            "portfolio": "https://refaj-hossan.vercel.app/",
        },
        # Add more collaborators as needed
    ]

    for collaborator_info in collaborators_info:
        st.write(f"## {collaborator_info['name']}")
        
        st.write(f"**Github Profile:** [![GitHub](https://img.shields.io/badge/GitHub-Profile-brightgreen?style=flat-square&logo=github)]({collaborator_info['github']})")
        st.write(f"**Linkedin Account:** [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat-square&logo=linkedin)]({collaborator_info['linkedin']})")
        st.write(f"**Portfolio:** [![Portfolio](https://img.shields.io/badge/Portfolio-Website-orange?style=flat-square&logo=vercel)]({collaborator_info['portfolio']})")

        st.markdown("---")

# Section: Recommendations
elif selected_section == "Recommendations":
    st.title("Recommendations")
    st.write("We welcome future collaborations and suggestions for improving this project. ")
    st.write("### Encouraging Continuous Improvement")
    st.write("As we embark on this venture, our motivation extends to fostering a community of learners and practitioners. We invite  feedback, suggestions, "
             "and collaboration to continuously improve our models and contribute to the collective knowledge in the field of handwritten digit prediction.")

    st.write("Join us on this exciting journey of pushing the boundaries of MNIST-based handwritten number prediction and realizing the practical impact "
             "it can have on various applications!")

    st.write("Contact:")
    st.write("- Email: tofayelahmmedbabu@gmail.com or mdrefajhossan@example.com")
    st.write("- LinkedIn: [tofayelahmmedbabu](https://www.linkedin.com/in/tofayelahmmedbabu/) or [refajhossan](https://www.linkedin.com/in/tofayelahmmedbabu/) ")
    
