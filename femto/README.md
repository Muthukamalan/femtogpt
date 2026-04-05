### Dataset Viewer
```jsonc
// Cosmopedia
{
  "prompt": "Here is an extract from a webpage:\n\"Ruangguru Free Online Schools to Support Students and Teachers\nRuangguru re-opened Ruangguru Online School for free through its application. This is a form and effort of Ruangguru in responding to the high need for distance learning facilities (PJJ) during the Covid-19 pandemic.\nPreviously, since March 2020 when learning activities from home were implemented due to Covid-19, Ruangguru has launched the free Ruangguru Online School.\nIn the past school year, the Ruangguru Online School received a positive response with more than 5,000,000 students using it. More than 100,000 teachers have also participated in free online teacher training provided on the Ruangguru app.\nThe enthusiasm of the teachers and students to continue learning inspires us to re-open the Ruangguru Free Online School, “explained Ruangguru’s founder and Chief Director, Belva Devara, as quoted from his official statement on Monday (3/8/2020).\nRuangguru Online School 2 for Free\nThrough the Ruangguru 2 Online School progra.\"\n\nWrite an extensive and detailed course unit in a textbook for college students related to the extract above, focusing on \"Social Studies Integration\" as part of \"EDUCATION / Schools / Levels / Elementary\". Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth.\n\nTry to:\n\n- Ensure in-depth coverage of the concepts.\n- Use a narrative, thought-provoking style.\n- Connect the topic with current trends, real-life examples, or recent studies.\nDo not include a title, introductory phrases, or images. Write the content directly.",
  "text": " In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. As such, social studies integration in elementary education can greatly benefit from incorporating digital tools like those offered by Ruangguru's Online School. Let's explore how educators can effectively leverage these resources to create engaging and meaningful learning experiences for young students.\n\nFirstly, let's define what we mean by social studies integration. Social studies encompasses various disciplines that help students understand their communities, societies, and the wider world around them. These subjects may include history, geography, civics, economics, sociology, and anthropology. By integrating social studies into the curriculum, we aim to foster critical thinking skills, promote cultural awareness, and encourage responsible citizenship among students. Now, let's see how Ruangguru's Online School supports these goals.\n\nRuangguru's Online School offers a wealth of opportunities for elementary students to engage with social studies topics in a dynamic and interactive manner. For instance, instead of simply reading about historical events or figures, students can participate in virtual simulations where they assume roles within specific time periods or cultures. Such immersive experiences allow children to connect emotionally with the material while developing essential collaboration and problem-solving abilities.\n\nAnother significant advantage of utilizing Ruangguru's platform is accessibility. With many schools closed due to the ongoing health crisis, ensuring continuity of learning has become paramount. Through Ruangguru's mobile application, students can access quality instructional materials anytime, anywhere—an especially crucial feature for families without reliable internet connections at home. Moreover, the platform allows teachers to track student progress, enabling them to provide targeted support and intervention when necessary.\n\nFurthermore, Ruangguru recognizes the importance of professional development for educators transitioning to remote teaching environments. To facilitate this process, they offer complimentary online teacher training sessions via their app. During these workshops, instructors can share best practices, discuss challenges, and collaborate on innovative strategies for delivering effective social studies lessons in a virtual setting. Additionally, connecting with fellow professionals fosters a sense of community among teachers, helping them feel supported and valued during uncertain times.\n\nMoreover, Ruangguru's Online School aligns seamlessly with current trends emphasizing project-based learning (PBL), which encourages students to apply knowledge and skills to solve authentic problems. For example, students might research local environmental issues, analyze data, and propose solutions using multimedia presentations created within the app. PBL approaches not only deepen conceptual understanding but also empower students to take action and contribute positively to their communities.\n\nLastly, Ruangguru's commitment to providing equitable access to quality education resonates strongly with recent calls for social justice reform in education. By offering free resources and training, they democratize learning opportunities, allowing all students to thrive regardless of socioeconomic background. Furthermore, incorporating diverse perspectives and histories into social studies curricula promotes inclusivity and empathy, preparing future generations to navigate an increasingly interconnected global society.\n\nIn conclusion, harnessing the power of digital platforms like Ruangguru's Online School can significantly enhance social studies integration efforts at the elementary level. Amidst unprecedented challenges brought forth by the COVID-19 pandemic, embracing innovation becomes vital in creating resilient learners who are well-equipped to tackle complex issues facing our world today. Together, let us strive towards cultivating engaged, informed, and compassionate citizens ready to make a difference in their communities and beyond.",
  "token_length": 805,
  "audience": "college_students",
  "format": "textbook",
  "seed_data": "fineweb"
}
// databricks/dolly-15k
{
  "instruction": "When did Virgin Australia start operating?",
  "context": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.",
  "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
  "category": "closed_qa"
}
// Abirate/english_quotes
{
  "quote": "Be yourself; everyone else is already taken.",
  "author": "Oscar Wilde",
  "tags": [
    "be-yourself",
    "gilbert-perreira",
    "honesty",
    "inspirational",
    "misattributed-oscar-wilde",
    "quote-investigator"
  ]
}

// b-mc2/sql-create-context
{
  "answer": "SELECT COUNT(*) FROM head WHERE age > 56",
  "question": "How many heads of the departments are older than 56 ?",
  "context": "CREATE TABLE head (age INTEGER)"
}

// Squad
{
  "id": "5733be284776f41900661182",
  "title": "University_of_Notre_Dame",
  "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
  "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
  "answers": {
    "text": ["Saint Bernadette Soubirous"],
    "answer_start": [515]
  }
}

// alpaca
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.",
  "text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
}

```