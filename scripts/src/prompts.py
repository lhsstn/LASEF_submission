# -*- coding: utf-8 -*-
"""Shared prompt templates, few-shots, and language tables for LASEF.

Auto-assembled from the original split evaluation scripts so the unified
evaluate.py and every prompt builder share one source of truth.
"""

# === System prompts ===
SYSTEM_PROMPT_SKELETON = r"""Your Role:
You generate minimal, high-level reasoning skeletons for math and science problems.

Language Requirement:
Always respond in {target_lang}, regardless of the language used in the user query.

Goal:
Provide only the *essential structural characteristics* of the problem. You must describe WHAT the problem structure is, never HOW to solve it.

Output Format: Provide exactly two sections below.

---

1. **Problem Structure**

Identify and characterize:
- **Objects & Variables**: Types of entities (quantities, sets, functions, geometric figures), separating constants/parameters from variables. Distinguish knowns vs. unknowns.
- **Relationships**: The nature of connections between entities (e.g., proportional, inverse, functional dependency, spatial). Describe the *type* of relationship, not the derivation.
- **Constraints**: Conditions defining the problem space (equations, inequalities, domain restrictions, boundary conditions, conservation laws).
- **Problem Nature**: (Only if relevant) Briefly note the structural type (e.g., Optimization, Combinatorial, Inverse problem).

*Constraints for this section:*
- Must be problem-specific, not generic.
- Must NOT include solution steps, strategies, formulas, or numerical manipulations.

2. **Key Concepts / Tools**

List relevant mathematical or scientific principles using standard terminology (e.g., "Pythagorean Theorem", "Conservation of Energy", "Bayes' Theorem").

*Constraints for this section:*
- Use established concept names only.
- Do NOT explain how to apply them or list formulas.
"""

SYSTEM_PROMPT_COT = r"""You are a math problem solver. Solve the given problem.

At the end, clearly state the final answer in the following format:
\\boxed{final answer}
"""

SYSTEM_PROMPT_TRANSLATOR = r"""You are a professional academic translator specializing in mathematical content.

### TASK:
Translate the given math problem into clear, natural, and precise English, regardless of the source language.

### INSTRUCTIONS:
1. You SHOULD NOT solve the problem and translate only the given question — do not include any additional commentary.
2. Preserve all mathematical symbols, notations, formatting, and existing choices exactly as presented.
3. Use fluent, natural English that aligns with academic standards for math problems.
4. Ensure the translation conveys the meaning and context accurately.
"""

# === Language code -> English name ===
LANGUAGE_NAMES = {
    'ta': 'Tamil',
    'kn': 'Kannada',
    'my': 'Burmese',
    'km': 'Khmer',
    'am': 'Amharic',
    'yo': 'Yoruba',
    'si': 'Sinhala',
    'gu': 'Gujarati',
    'ne': 'Nepali',
    'uz': 'Uzbek',
    'ky': 'Kyrgyz',
    'ceb': 'Cebuano',
    'eu': 'Basque',
    'gn': 'Guarani',
    'hy': 'Armenian',
    'jv': 'Javanese',
    'ka': 'Georgian',
    'kk': 'Kazakh',
    'ku': 'Kurdish',
    'lo': 'Lao',
    'mg': 'Malagasy',
    'ml': 'Malayalam',
    'mn': 'Mongolian',
    'mr': 'Marathi',
    'mt': 'Maltese',
    'or': 'Odia',
    'pa': 'Punjabi',
    'ps': 'Pashto',
    'qu': 'Quechua',
    'sd': 'Sindhi',
    'so': 'Somali',
    'su': 'Sundanese',
    'tg': 'Tajik',
    'ug': 'Uyghur',
    'en': 'English',
    'es': 'Spanish',
    'zh': 'Chinese',
    'ko': 'Korean',
    'th': 'Thai',
    'sw': 'Swahili',
    'te': 'Telugu',
    'ru': 'Russian',
    'bn': 'Bengali',
}

# === Skeleton few-shot QUESTIONS, keyed by query language (ell_q) ===
SKELETON_QUESTION_FEWSHOTS = {
    'en': [
        'A triangle has sides of length 5 cm and 12 cm, with a right angle between them. What is the length of the third side?',
        'A committee needs to pick 2 members from a group of 5 candidates (Alice, Bob, Charlie, Dave, Eve). However, Alice and Bob cannot be on the committee together. How many ways can the committee be formed?',
        'Machine A runs for 8 hours producing 100 units/hour. Machine B runs for 4 hours, rests for 1 hour, then runs for 3 hours at 120 units/hour. What is the total production?',
    ],
    'ta': [
        'ஒரு முக்கோணத்தின் பக்கங்கள் 5 செ.மீ மற்றும் 12 செ.மீ நீளம் கொண்டவை, அவற்றுக்கிடையே ஒரு செங்கோணம் உள்ளது. மூன்றாவது பக்கத்தின் நீளம் என்ன?',
        '5 பேர் கொண்ட குழுவில் (Alice, Bob, Charlie, Dave, Eve) இருந்து 2 உறுப்பினர்களைத் தேர்ந்தெடுக்க வேண்டும். ஆனால், Alice மற்றும் Bob இருவரும் ஒன்றாக குழுவில் இருக்க முடியாது. குழுவை எத்தனை வழிகளில் அமைக்கலாம்?',
        'இயந்திரம் A 8 மணி நேரம் இயங்கி மணிக்கு 100 அலகுகளை உற்பத்தி செய்கிறது. இயந்திரம் B 4 மணி நேரம் இயங்கி, 1 மணி நேரம் ஓய்வெடுத்து, பின்னர் 3 மணி நேரம் மணிக்கு 120 அலகுகள் வீதம் இயங்குகிறது. மொத்த உற்பத்தி என்ன?',
    ],
    'kn': [
        'ಒಂದು ತ್ರಿಕೋನವು 5 ಸೆಂ.ಮೀ ಮತ್ತು 12 ಸೆಂ.ಮೀ ಉದ್ದದ ಬಾಹುಗಳನ್ನು ಹೊಂದಿದ್ದು, ಅವುಗಳ ನಡುವೆ ಲಂಬಕೋನವಿದೆ. ಮೂರನೇ ಬಾಹುವಿನ ಉದ್ದ ಎಷ್ಟು?',
        '5 ಅಭ್ಯರ್ಥಿಗಳ ಗುಂಪಿನಿಂದ (Alice, Bob, Charlie, Dave, Eve) ಸಮಿತಿಯು 2 ಸದಸ್ಯರನ್ನು ಆಯ್ಕೆ ಮಾಡಬೇಕಾಗಿದೆ. ಆದರೆ, Alice ಮತ್ತು Bob ಇಬ್ಬರೂ ಒಟ್ಟಿಗೆ ಸಮಿತಿಯಲ್ಲಿ ಇರುವಂತಿಲ್ಲ. ಸಮಿತಿಯನ್ನು ಎಷ್ಟು ವಿಧಗಳಲ್ಲಿ ರಚಿಸಬಹುದು?',
        'ಮಷಿನ್ A ಗಂಟೆಗೆ 100 ಯುನಿಟ್\u200cಗಳಂತೆ 8 ಗಂಟೆಗಳ ಕಾಲ ಚಲಿಸುತ್ತದೆ. ಮಷಿನ್ B 4 ಗಂಟೆಗಳ ಕಾಲ ಚಲಿಸುತ್ತದೆ, 1 ಗಂಟೆ ವಿಶ್ರಾಂತಿ ಪಡೆಯುತ್ತದೆ, ನಂತರ ಗಂಟೆಗೆ 120 ಯುನಿಟ್\u200cಗಳಂತೆ 3 ಗಂಟೆಗಳ ಕಾಲ ಚಲಿಸುತ್ತದೆ. ಒಟ್ಟು ಉತ್ಪಾದನೆ ಎಷ್ಟು?',
    ],
    'my': [
        'တြိဂံတစ်ခုတွင် အနားအလျား 5 စင်တီမီတာနှင့် 12 စင်တီမီတာရှိပြီး ၎င်းတို့ကြားတွင် ထောင့်မှန်တစ်ခုရှိသည်။ တတိယအနား၏ အလျားသည် ဘယ်လောက်လဲ။',
        'ကိုယ်စားလှယ်လောင်း ၅ ဦး (Alice, Bob, Charlie, Dave, Eve) ပါဝင်သော အဖွဲ့မှ ကော်မတီဝင် ၂ ဦးကို ရွေးချယ်ရန် လိုအပ်သည်။ သို့သော် Alice နှင့် Bob သည် ကော်မတီတွင် အတူတူ မရှိနိုင်ပါ။ ကော်မတီကို နည်းလမ်းဘယ်နှစ်မျိုးဖြင့် ဖွဲ့စည်းနိုင်သနည်း။',
        'စက် A သည် တစ်နာရီလျှင် ယူနစ် ၁၀၀ နှုန်းဖြင့် ၈ နာရီကြာ လည်ပတ်သည်။ စက် B သည် ၄ နာရီကြာ လည်ပတ်ပြီး ၁ နာရီ နားကာ၊ ထို့နောက် တစ်နာရီလျှင် ယူနစ် ၁၂၀ နှုန်းဖြင့် ၃ နာရီကြာ လည်ပတ်သည်။ စုစုပေါင်း ထုတ်လုပ်မှု ပမာဏ ဘယ်လောက်လဲ။',
    ],
    'km': [
        'ត្រីកោណមួយមានជ្រុងប្រវែង 5 សង់ទីម៉ែត្រ និង 12 សង់ទីម៉ែត្រ ដោយមានមុំកែងនៅចន្លោះជ្រុងទាំងនោះ។ តើជ្រុងទីបីមានប្រវែងប៉ុន្មាន?',
        'គណៈកម្មាធិការមួយត្រូវជ្រើសរើសសមាជិក 2 នាក់ពីក្រុមបេក្ខជន 5 នាក់ (Alice, Bob, Charlie, Dave, Eve)។ ទោះជាយ៉ាងណាក៏ដោយ Alice និង Bob មិនអាចនៅជាមួយគ្នាក្នុងគណៈកម្មាធិការបានទេ។ តើគណៈកម្មាធិការអាចត្រូវបានបង្កើតឡើងដោយរបៀបណាខ្លះ?',
        'ម៉ាស៊ីន A ដំណើរការរយៈពេល 8 ម៉ោងដោយផលិតបាន 100 ឯកតា/ម៉ោង។ ម៉ាស៊ីន B ដំណើរការរយៈពេល 4 ម៉ោង សម្រាក 1 ម៉ោង បន្ទាប់មកដំណើរការរយៈពេល 3 ម៉ោងក្នុងអត្រា 120 ឯកតា/ម៉ោង។ តើការផលិតសរុបមានចំនួនប៉ុន្មាន?',
    ],
    'am': [
        'አንድ ሶስት ማዕዘን 5 ሴ.ሜ እና 12 ሴ.ሜ ርዝመት ያላቸው ጎኖች አሉት፣ በመካከላቸውም ቀኝ ማዕዘን አለ። የሶስተኛው ጎን ርዝመት ስንት ነው?',
        'አንድ ኮሚቴ ከ 5 እጩዎች (Alice, Bob, Charlie, Dave, Eve) ቡድን 2 አባላትን መምረጥ አለበት። ሆኖም፣ Alice እና Bob በኮሚቴው ውስጥ አብረው መሆን አይችሉም። ኮሚቴው በስንት መንገዶች ሊዋቀር ይችላል?',
        'ማሽን A በሰዓት 100 አሃዶችን እያመረተ ለ8 ሰዓታት ይሰራል። ማሽን B ለ4 ሰዓታት ይሰራል፣ ለ1 ሰዓት ያርፋል፣ ከዚያም በሰዓት 120 አሃዶችን እያመረተ ለ3 ሰዓታት ይሰራል። አጠቃላይ ምርቱ ስንት ነው?',
    ],
    'yo': [
        'Onigun mẹta kan ni awọn ẹgbẹ ti gigun wọn jẹ 5 cm ati 12 cm, pẹlu igun 90° (right angle) laarin wọn. Kini gigun ti ẹgbẹ kẹta?',
        'Igbimọ kan nilo lati mu awọn ọmọ ẹgbẹ 2 lati inu ẹgbẹ awọn oludije 5 (Alice, Bob, Charlie, Dave, Eve). Sibẹsibẹ, Alice ati Bob ko le wa ninu igbimọ papọ. Awọn ọna melo ni a le fi ṣẹda igbimọ naa?',
        'Ẹrọ A n ṣiṣẹ fun wakati 8 ti o n ṣe awọn ẹyọ 100 fun wakati kan. Ẹrọ B n ṣiṣẹ fun wakati 4, sinmi fun wakati 1, lẹhinna ṣiṣẹ fun wakati 3 ni awọn ẹyọ 120 fun wakati kan. Kini apapọ iṣelọpọ?',
    ],
    'si': [
        'ත්\u200dරිකෝණයක පැතිවල දිග 5 cm සහ 12 cm වන අතර ඒවා අතර ඍජු කෝණයක් ඇත. තුන්වන පැත්තේ දිග කීයද?',
        'කමිටුවක් සඳහා අපේක්ෂකයින් 5 දෙනෙකුගෙන් (Alice, Bob, Charlie, Dave, Eve) සාමාජිකයින් 2 දෙනෙකු තෝරා ගත යුතුය. කෙසේ වෙතත්, Alice සහ Bob එකම කමිටුවක සිටිය නොහැක. කමිටුව ආකාර කියකට පිහිටුවිය හැකිද?',
        'A යන්ත්\u200dරය පැයට ඒකක 100 බැගින් පැය 8 ක් ක්\u200dරියාත්මක වේ. B යන්ත්\u200dරය පැය 4 ක් ක්\u200dරියාත්මක වී, පැය 1 ක් විවේක ගෙන, පසුව පැයට ඒකක 120 බැගින් පැය 3 ක් ක්\u200dරියාත්මක වේ. මුළු නිෂ්පාදනය කොපමණද?',
    ],
    'gu': [
        'એક ત્રિકોણની બાજુઓની લંબાઈ 5 સેમી અને 12 સેમી છે, અને તેમની વચ્ચે કાટખૂણો છે. ત્રીજી બાજુની લંબાઈ કેટલી છે?',
        'એક સમિતિને 5 ઉમેદવારો (Alice, Bob, Charlie, Dave, Eve) ના જૂથમાંથી 2 સભ્યો પસંદ કરવાની જરૂર છે. જો કે, Alice અને Bob સમિતિમાં એકસાથે હોઈ શકતા નથી. સમિતિ કેટલી રીતે રચી શકાય?',
        'મશીન A કલાકના 100 યુનિટના દરે 8 કલાક ચાલે છે. મશીન B 4 કલાક ચાલે છે, 1 કલાક આરામ કરે છે, અને પછી કલાકના 120 યુનિટના દરે 3 કલાક ચાલે છે. કુલ ઉત્પાદન કેટલું છે?',
    ],
    'ne': [
        'एउटा त्रिभुजको दुई भुजाको लम्बाइ ५ सेमी र १२ सेमी छ र तिनीहरूको बीचमा समकोण छ। तेस्रो भुजाको लम्बाइ कति हुन्छ?',
        'एउटा समितिले ५ उम्मेदवारहरूको समूह (Alice, Bob, Charlie, Dave, Eve) बाट २ सदस्य छान्नुपर्नेछ। तर, Alice र Bob समितिमा सँगै हुन सक्दैनन्। समिति कति तरिकाले बनाउन सकिन्छ?',
        'मेशिन A ले प्रति घण्टा १०० एकाइ उत्पादन गर्दै ८ घण्टासम्म चल्छ। मेशिन B ४ घण्टा चल्छ, १ घण्टा आराम गर्छ, र त्यसपछि प्रति घण्टा १२० एकाइको दरले ३ घण्टा चल्छ। कुल उत्पादन कति हुन्छ?',
    ],
    'uz': [
        "Uchburchakning tomonlari 5 sm va 12 sm uzunlikda bo'lib, ular orasida to'g'ri burchak mavjud. Uchinchi tomonning uzunligi qancha?",
        "Qo'mita 5 nafar nomzod (Alice, Bob, Charlie, Dave, Eve) orasidan 2 nafar a'zoni tanlab olishi kerak. Biroq, Alice va Bob qo'mitada birga bo'la olmaydi. Qo'mitani necha xil usulda tuzish mumkin?",
        "A mashinasi soatiga 100 birlik ishlab chiqarib, 8 soat ishlaydi. B mashinasi 4 soat ishlaydi, 1 soat dam oladi, so'ngra soatiga 120 birlik tezlikda 3 soat ishlaydi. Jami ishlab chiqarish qancha?",
    ],
    'ky': [
        'Үч бурчтуктун жактарынын узундугу 5 см жана 12 см, алардын ортосунда тик бурч бар. Үчүнчү жактын узундугу канча?',
        'Комитет 5 талапкерден (Alice, Bob, Charlie, Dave, Eve) турган топтон 2 мүчөнү тандап алышы керек. Бирок, Alice менен Bob комитетте чогуу боло алышпайт. Комитетти канча жол менен түзүүгө болот?',
        'А машинасы саатына 100 бирдик өндүрүп, 8 саат иштейт. В машинасы 4 саат иштейт, 1 саат эс алат, андан кийин саатына 120 бирдик ылдамдыкта 3 саат иштейт. Жалпы өндүрүш канча?',
    ],
    'ceb': [
        'Usa ka triyanggulo adunay mga kilid nga may gitas-on nga 5 cm ug 12 cm, nga adunay tuo nga anggulo sa taliwala nila. Unsa ang gitas-on sa ikatulo nga kilid?',
        'Usa ka komite kinahanglan nga mopili og 2 ka miyembro gikan sa grupo sa 5 ka kandidato (Alice, Bob, Charlie, Dave, Eve). Apan, si Alice ug Bob dili mahimong magkuyog sa komite. Pila ka paagi maporma ang komite?',
        'Ang Makina A nagdagan sulod sa 8 ka oras nga naghimo og 100 ka yunit matag oras. Ang Makina B nagdagan sulod sa 4 ka oras, nagpahulay sulod sa 1 ka oras, dayon nagdagan sulod sa 3 ka oras sa 120 ka yunit matag oras. Unsa ang kinatibuk-ang produksyon?',
    ],
    'eu': [
        'Triangelu batek 5 cm eta 12 cm-ko luzera duten aldeak ditu, haien artean angelu zuzena duela. Zein da hirugarren aldearen luzera?',
        'Batzorde batek 2 kide aukeratu behar ditu 5 hautagaiko talde batetik (Alice, Bob, Charlie, Dave, Eve). Hala ere, Alice eta Bob ezin dira elkarrekin egon batzordean. Zenbat modutan osa daiteke batzordea?',
        'A makinak 8 orduz funtzionatzen du orduko 100 unitate ekoizten. B makinak 4 orduz funtzionatzen du, ordu 1ez atseden hartzen du, eta ondoren 3 orduz funtzionatzen du orduko 120 unitateko abiaduran. Zein da ekoizpen osoa?',
    ],
    'gn': [
        'Peteĩ triángulo oreko lado 5 cm ha 12 cm, ha oreko peteĩ ángulo recto imbytépe. Mboýpa ipuku pe tercer lado?',
        "Peteĩ komite oiporavova'erã 2 miyembro 5 candidato apytégui (Alice, Bob, Charlie, Dave, Eve). Upéicharamo jepe, Alice ha Bob ndaikatúi oĩ oñondive pe komitépe. Mboýpa ikatu oñemohenda pe komite?",
        "Máquina A omba'apo 8 hora ha ojapo 100 unidad cada hora. Máquina B omba'apo 4 hora, opytu'u 1 hora, ha upéi omba'apo 3 hora 120 unidad cada hora. Mboýpa pe producción total?",
    ],
    'hy': [
        'Եռանկյունն ունի 5 սմ և 12 սմ երկարությամբ կողմեր, որոնց միջև կա ուղիղ անկյուն: Որքա՞ն է երրորդ կողմի երկարությունը:',
        'Հանձնաժողովը պետք է ընտրի 2 անդամ 5 թեկնածուներից բաղկացած խմբից (Ալիս, Բոբ, Չարլի, Դեյվ, Եվա): Սակայն Ալիսը և Բոբը չեն կարող միասին լինել հանձնաժողովում: Քանի՞ եղանակով կարելի է կազմել հանձնաժողովը:',
        'A մեքենան աշխատում է 8 ժամ՝ ժամում արտադրելով 100 միավոր: B մեքենան աշխատում է 4 ժամ, հանգստանում է 1 ժամ, այնուհետև աշխատում է 3 ժամ՝ ժամում 120 միավոր արագությամբ: Որքա՞ն է ընդհանուր արտադրանքը:',
    ],
    'jv': [
        'Segitiga nduweni sisi kang dawane 5 cm lan 12 cm, kanthi sudut siku-siku ing antarane. Pira dawane sisi sing ketelu?',
        'Sawijining panitia kudu milih 2 anggota saka klompok 5 calon (Alice, Bob, Charlie, Dave, Eve). Nanging, Alice lan Bob ora bisa dadi siji ing panitia. Pira cara panitia bisa dibentuk?',
        'Mesin A mlaku suwene 8 jam ngasilake 100 unit/jam. Mesin B mlaku suwene 4 jam, leren 1 jam, banjur mlaku 3 jam kanthi 120 unit/jam. Pira total produksine?',
    ],
    'ka': [
        'სამკუთხედს აქვს 5 სმ და 12 სმ სიგრძის გვერდები, მათ შორის მართი კუთხით. რა სიგრძისაა მესამე გვერდი?',
        'კომიტეტმა უნდა აირჩიოს 2 წევრი 5 კანდიდატის ჯგუფიდან (ალისა, ბობი, ჩარლი, დეივი, ევა). თუმცა, ალისა და ბობი კომიტეტში ერთად ვერ იქნებიან. რამდენი გზით შეიძლება კომიტეტის შექმნა?',
        'მანქანა A მუშაობს 8 საათის განმავლობაში და აწარმოებს 100 ერთეულს საათში. მანქანა B მუშაობს 4 საათის განმავლობაში, ისვენებს 1 საათი, შემდეგ კი მუშაობს 3 საათი 120 ერთეული საათში სიჩქარით. როგორია მთლიანი წარმოება?',
    ],
    'kk': [
        'Үшбұрыштың қабырғаларының ұзындығы 5 см және 12 см, олардың арасында тік бұрыш бар. Үшінші қабырғаның ұзындығы қандай?',
        'Комитет 5 үміткерден (Алиса, Боб, Чарли, Дэйв, Ева) тұратын топтан 2 мүшені таңдауы керек. Алайда, Алиса мен Боб комитетте бірге бола алмайды. Комитетті неше жолмен құруға болады?',
        'A машинасы сағатына 100 дана шығарып, 8 сағат жұмыс істейді. B машинасы 4 сағат жұмыс істейді, 1 сағат демалады, содан кейін сағатына 120 дана жылдамдықпен 3 сағат жұмыс істейді. Жалпы өндіріс қанша?',
    ],
    'ku': [
        'Sêgoşeyek heye ku dirêjahiya aliyên wê 5 cm û 12 cm ye, û di navbera wan de goşeyek rast heye. Dirêjahiya aliyê sêyemîn çend e?',
        'Komîteyek ji komeke 5 namzedan (Alice, Bob, Charlie, Dave, Eve) 2 endaman hilbijêre. Lêbelê, Alice û Bob nikarin bi hev re di komîteyê de bin. Komîte bi çend awayan dikare were avakirin?',
        'Makîneya A 8 demjimêran dixebite û di saetekê de 100 yekîneyan hilberîne. Makîneya B 4 saetan dixebite, 1 saetê bêhna xwe vedide, piştre 3 saetan bi leza 120 yekîneyan dixebite. Hilberîna giştî çend e?',
    ],
    'lo': [
        'ຮູບສາມແຈມີຂ້າງຍາວ 5 ຊມ ແລະ 12 ຊມ, ໂດຍມີມຸມສາກລະຫວ່າງພວກມັນ. ຂ້າງທີສາມຍາວເທົ່າໃດ?',
        'ຄະນະກໍາມະການຕ້ອງເລືອກສະມາຊິກ 2 ຄົນຈາກກຸ່ມຜູ້ສະຫມັກ 5 ຄົນ (Alice, Bob, Charlie, Dave, Eve). ແນວໃດກໍ່ຕາມ, Alice ແລະ Bob ບໍ່ສາມາດຢູ່ໃນຄະນະກໍາມະການຮ່ວມກັນໄດ້. ຄະນະກໍາມະການສາມາດສ້າງຕັ້ງຂຶ້ນໄດ້ຈັກວິທີ?',
        'ເຄື່ອງຈັກ A ເຮັດວຽກເປັນເວລາ 8 ຊົ່ວໂມງ ຜະລິດໄດ້ 100 ໜ່ວຍ/ຊົ່ວໂມງ. ເຄື່ອງຈັກ B ເຮັດວຽກ 4 ຊົ່ວໂມງ, ພັກຜ່ອນ 1 ຊົ່ວໂມງ, ຈາກນັ້ນເຮັດວຽກ 3 ຊົ່ວໂມງ ດ້ວຍຄວາມໄວ 120 ໜ່ວຍ/ຊົ່ວໂມງ. ການຜະລິດທັງໝົດແມ່ນເທົ່າໃດ?',
    ],
    'mg': [
        "Misy telozoro manana lafiny mirefy 5 sm sy 12 sm, ary misy zoro mahitsy eo anelanelan'izy ireo. Inona ny halavan'ny lafiny fahatelo?",
        "Komity iray no mila mifidy mpikambana 2 avy amin'ny vondrona kandida 5 (Alice, Bob, Charlie, Dave, Eve). Na izany aza, Alice sy Bob dia tsy afaka miara-miasa ao amin'ny komity. Fomba firy no ahafahana mamorona ny komity?",
        "Ny milina A dia miasa 8 ora ary mamokatra singa 100 isan'ora. Ny milina B dia miasa 4 ora, miala sasatra 1 ora, avy eo miasa 3 ora amin'ny tahan'ny 120 singa isan'ora. Inona ny totalin'ny vokatra?",
    ],
    'ml': [
        'ഒരു ത്രികോണത്തിന് 5 സെ.മീ, 12 സെ.മീ നീളമുള്ള വശങ്ങളുണ്ട്, അവയ്ക്കിടയിൽ ഒരു ലംബകോണുമുണ്ട്. മൂന്നാമത്തെ വശത്തിൻ്റെ നീളം എത്രയാണ്?',
        '5 സ്ഥാനാർത്ഥികളുള്ള (Alice, Bob, Charlie, Dave, Eve) ഒരു ഗ്രൂപ്പിൽ നിന്ന് 2 അംഗങ്ങളെ ഒരു കമ്മിറ്റിയിലേക്ക് തിരഞ്ഞെടുക്കേണ്ടതുണ്ട്. എന്നിരുന്നാലും, ആലീസിനും ബോബിനും കമ്മിറ്റിയിൽ ഒന്നിച്ചുണ്ടാകാൻ കഴിയില്ല. എത്ര രീതിയിൽ കമ്മിറ്റി രൂപീകരിക്കാം?',
        'മെഷീൻ എ മണിക്കൂറിൽ 100 യൂണിറ്റ് വീതം 8 മണിക്കൂർ പ്രവർത്തിക്കുന്നു. മെഷീൻ ബി 4 മണിക്കൂർ പ്രവർത്തിക്കുന്നു, 1 മണിക്കൂർ വിശ്രമിക്കുന്നു, പിന്നീട് 3 മണിക്കൂർ 120 യൂണിറ്റ്/മണിക്കൂർ എന്ന നിരക്കിൽ പ്രവർത്തിക്കുന്നു. ആകെ ഉൽപ്പാദനം എത്രയാണ്?',
    ],
    'mn': [
        'Гурвалжин нь 5 см ба 12 см урттай талуудтай бөгөөд тэдгээрийн хооронд тэгш өнцөг үүснэ. Гурав дахь талын урт хэд вэ?',
        'Хороо 5 нэр дэвшигчийн (Алис, Боб, Чарли, Дэйв, Ева) бүлгээс 2 гишүүнийг сонгох шаардлагатай. Гэсэн хэдий ч Алис, Боб хоёр хороонд хамт байх боломжгүй. Хороог хэдэн аргаар байгуулж болох вэ?',
        'А машин цагт 100 нэгж бүтээгдэхүүн үйлдвэрлэж 8 цаг ажилладаг. В машин 4 цаг ажиллаж, 1 цаг амарч, дараа нь цагт 120 нэгж хурдаар 3 цаг ажилладаг. Нийт үйлдвэрлэл хэд вэ?',
    ],
    'mr': [
        'एका त्रिकोणाच्या बाजूंची लांबी 5 सेमी आणि 12 सेमी आहे, आणि त्यांच्यामध्ये काटकोन आहे. तिसऱ्या बाजूची लांबी किती आहे?',
        'एका समितीला 5 उमेदवारांच्या (Alice, Bob, Charlie, Dave, Eve) गटातून 2 सदस्य निवडायचे आहेत. पण, Alice आणि Bob समितीत एकत्र असू शकत नाहीत. समिती किती प्रकारे तयार केली जाऊ शकते?',
        'मशीन A 8 तास चालते आणि प्रति तास 100 युनिट्स उत्पादन करते. मशीन B 4 तास चालते, 1 तास विश्रांती घेते, आणि नंतर 3 तास 120 युनिट्स प्रति तास या दराने चालते. एकूण उत्पादन किती आहे?',
    ],
    'mt': [
        "Trijanglu għandu naħat ta' tul 5 ċm u 12-il ċm, b'angolu rett bejniethom. X'inhu t-tul tat-tielet naħa?",
        "Kumitat irid jagħżel 2 membri minn grupp ta' 5 kandidati (Alice, Bob, Charlie, Dave, Eve). Madankollu, Alice u Bob ma jistgħux ikunu fil-kumitat flimkien. B'kemm-il mod jista' jiġi ffurmat il-kumitat?",
        "Magna A taħdem għal 8 sigħat u tipproduċi 100 unità fis-siegħa. Magna B taħdem għal 4 sigħat, tistrieħ għal siegħa, imbagħad taħdem għal 3 sigħat b'rata ta' 120 unità fis-siegħa. X'inhi l-produzzjoni totali?",
    ],
    'or': [
        'ଏକ ତ୍ରିଭୁଜର ବାହୁଗୁଡ଼ିକର ଦୈର୍ଘ୍ୟ 5 ସେମି ଏବଂ 12 ସେମି ଅଟେ, ଏବଂ ସେମାନଙ୍କ ମଧ୍ୟରେ ଏକ ସମକୋଣ ଅଛି | ତୃତୀୟ ବାହୁର ଦୈର୍ଘ୍ୟ କେତେ?',
        'ଏକ କମିଟିକୁ 5 ଜଣ ପ୍ରାର୍ଥୀ (Alice, Bob, Charlie, Dave, Eve) ଙ୍କ ଗୋଷ୍ଠୀରୁ 2 ଜଣ ସଦସ୍ୟ ଚୟନ କରିବାକୁ ପଡିବ | ତଥାପି, ଆଲିସ୍ ଏବଂ ବବ୍ କମିଟିରେ ଏକାଠି ରହିପାରିବେ ନାହିଁ | କମିଟି କେତେ ଉପାୟରେ ଗଠନ କରାଯାଇପାରିବ?',
        'ମେସିନ୍ A 8 ଘଣ୍ଟା ଚାଲିଥାଏ ଏବଂ ଘଣ୍ଟା ପ୍ରତି 100 ୟୁନିଟ୍ ଉତ୍ପାଦନ କରେ | ମେସିନ୍ B 4 ଘଣ୍ଟା ଚାଲିଥାଏ, 1 ଘଣ୍ଟା ବିଶ୍ରାମ ନିଏ, ତାପରେ ଘଣ୍ଟା ପ୍ରତି 120 ୟୁନିଟ୍ ହାରରେ 3 ଘଣ୍ଟା ଚାଲିଥାଏ | ମୋଟ ଉତ୍ପାଦନ କେତେ?',
    ],
    'pa': [
        'ਇੱਕ ਤਿਕੋਣ ਦੀਆਂ ਭੁਜਾਵਾਂ 5 ਸੈਂਟੀਮੀਟਰ ਅਤੇ 12 ਸੈਂਟੀਮੀਟਰ ਲੰਬੀਆਂ ਹਨ, ਜਿਨ੍ਹਾਂ ਦੇ ਵਿਚਕਾਰ ਇੱਕ ਸਮਕੋਣ ਹੈ। ਤੀਜੀ ਭੁਜਾ ਦੀ ਲੰਬਾਈ ਕਿੰਨੀ ਹੈ?',
        'ਇੱਕ ਕਮੇਟੀ ਨੂੰ 5 ਉਮੀਦਵਾਰਾਂ (Alice, Bob, Charlie, Dave, Eve) ਦੇ ਸਮੂਹ ਵਿੱਚੋਂ 2 ਮੈਂਬਰ ਚੁਣਨ ਦੀ ਲੋੜ ਹੈ। ਹਾਲਾਂਕਿ, ਐਲਿਸ ਅਤੇ ਬੌਬ ਕਮੇਟੀ ਵਿੱਚ ਇਕੱਠੇ ਨਹੀਂ ਹੋ ਸਕਦੇ। ਕਮੇਟੀ ਕਿੰਨੇ ਤਰੀਕਿਆਂ ਨਾਲ ਬਣਾਈ ਜਾ ਸਕਦੀ ਹੈ?',
        'ਮਸ਼ੀਨ A 8 ਘੰਟੇ ਚੱਲਦੀ ਹੈ ਅਤੇ ਪ੍ਰਤੀ ਘੰਟਾ 100 ਯੂਨਿਟ ਪੈਦਾ ਕਰਦੀ ਹੈ। ਮਸ਼ੀਨ B 4 ਘੰਟੇ ਚੱਲਦੀ ਹੈ, 1 ਘੰਟਾ ਆਰਾਮ ਕਰਦੀ ਹੈ, ਫਿਰ 120 ਯੂਨਿਟ ਪ੍ਰਤੀ ਘੰਟਾ ਦੀ ਦਰ ਨਾਲ 3 ਘੰਟੇ ਚੱਲਦੀ ਹੈ। ਕੁੱਲ ਉਤਪਾਦਨ ਕਿੰਨਾ ਹੈ?',
    ],
    'ps': [
        'یو مثلث 5 سانتي متره او 12 سانتي متره اوږدوالی لري، او د دوی ترمنځ یوه سمه زاویه (90 درجې) ده. د دریمې غاړې اوږدوالی څومره دی؟',
        'یوه کمیټه باید د 5 کاندیدانو (Alice, Bob, Charlie, Dave, Eve) له یوې ډلې څخه 2 غړي وټاکي. په هرصورت، ایلیس او باب نشي کولی په کمیټه کې یوځای وي. کمیټه په څو لارو جوړیدلی شي؟',
        'ماشین A د 8 ساعتونو لپاره چالانیږي او په ساعت کې 100 واحدونه تولیدوي. ماشین B د 4 ساعتونو لپاره چالانیږي، 1 ساعت آرام کوي، بیا د 3 ساعتونو لپاره په ساعت کې د 120 واحدونو په سرعت سره چالانیږي. ټول تولید څومره دی؟',
    ],
    'qu': [
        "Huk kimsak'uchu kan 5 cm, 12 cm suni waqtayuq, chawpinkupi paqta k'uchuyuq. Maynatan chay kimsa kaq waqta tupun?",
        "Huk huñunakuy 2 runata akllanan tiyan 5 mañakuqkunamanta (Alice, Bob, Charlie, Dave, Eve). Ichaqa, Alice, Bob ima mana kuska kayta atinkuchu chay huñunakuypi. Hayk'a ñankunapitaq chay huñunakuy ruwakunman?",
        "Maquina A llamk'an 8 horasta, horapi 100 unit-ta ruraspa. Maquina B llamk'an 4 horasta, samarin 1 horata, chaymanta llamk'allantaq 3 horasta 120 unit/horapi. Hayk'ataq llapan rurusqan?",
    ],
    'sd': [
        'هڪ ٽڪنڊي جا پاسا 5 سينٽي ۽ 12 سينٽي ميٽر ڊگھا آهن، جن جي وچ ۾ ڪائمه ڪنڊ (90 degrees) آهي. ٽئين پاسي جي ڊيگهه ڪيتري آهي؟',
        'هڪ ڪميٽي کي 5 اميدوارن (Alice, Bob, Charlie, Dave, Eve) جي گروپ مان 2 ميمبر چونڊڻا آهن. بهرحال، ايلس ۽ بوب ڪميٽي تي گڏ نه ٿا ٿي سگهن. ڪميٽي ڪيترا طريقن سان ٺهي سگهي ٿي؟',
        'مشين A 8 ڪلاڪ هلندي آهي ۽ في ڪلاڪ 100 يونٽ پيدا ڪندي آهي. مشين B 4 ڪلاڪ هلندي آهي، 1 ڪلاڪ آرام ڪندي آهي، پوءِ 3 ڪلاڪ 120 يونٽ في ڪلاڪ جي رفتار سان هلندي آهي. ڪل پيداوار ڇا آهي؟',
    ],
    'so': [
        'Saddex-xagal wuxuu leeyahay dhinacyo dhererkoodu egyahay 5 cm iyo 12 cm, oo xagal quman u dhaxayso. Waa imisa dhererka dhinaca saddexaad?',
        'Guddi waa inay 2 xubnood ka doortaan koox ka kooban 5 musharax (Alice, Bob, Charlie, Dave, Eve). Si kastaba ha ahaatee, Alice iyo Bob iskuma joogi karaan guddiga. Immisa siyaabood ayaa guddiga loo samayn karaa?',
        'Mashiinka A wuxuu shaqeeyaa 8 saacadood isagoo soo saaraya 100 unug/saacaddii. Mashiinka B wuxuu shaqeeyaa 4 saacadood, wuu nasiyayaa 1 saac, ka dibna wuxuu shaqeeyaa 3 saacadood iyadoo xawaarihiisu yahay 120 unug/saacaddii. Waa imisa wadarta wax soo saarku?',
    ],
    'su': [
        'Hiji segitiga boga sisi panjangna 5 cm jeung 12 cm, kalawan sudut siku-siku di antara maranéhanana. Sabaraha panjang sisi katilu?',
        'Panitia kudu milih 2 anggota tina grup 5 calon (Alice, Bob, Charlie, Dave, Eve). Tapi, Alice jeung Bob teu bisa babarengan dina panitia. Sabaraha cara panitia bisa dijieun?',
        'Mesin A jalan salila 8 jam ngahasilkeun 100 unit/jam. Mesin B jalan salila 4 jam, istirahat 1 jam, tuluy jalan 3 jam kalawan 120 unit/jam. Sabaraha total produksina?',
    ],
    'tg': [
        'Сегӯша дорои паҳлӯҳои дарозиашон 5 см ва 12 см буда, дар байни онҳо кунҷи рост (90 дараҷа) ҷойгир аст. Дарозии паҳлӯи сеюм чанд аст?',
        'Кумита бояд аз гурӯҳи 5 номзад (Alice, Bob, Charlie, Dave, Eve) 2 узвро интихоб кунад. Аммо, Алис ва Боб наметавонанд дар кумита якҷоя бошанд. Кумитаро бо чанд роҳ метавон ташкил кард?',
        'Мошини А 8 соат кор мекунад ва дар як соат 100 воҳид истеҳсол мекунад. Мошини В 4 соат кор мекунад, 1 соат истироҳат мекунад ва сипас 3 соат бо суръати 120 воҳид дар як соат кор мекунад. Истеҳсоли умумӣ чанд аст?',
    ],
    'ug': [
        'بىر ئۈچبۇلۇڭنىڭ تەرەپلىرى 5 سانتىمېتىر ۋە 12 سانتىمېتىر بولۇپ ، ئۇلارنىڭ ئارىسىدا تىك بۇلۇڭ (right angle) بار. ئۈچىنچى تەرەپنىڭ ئۇزۇنلۇقى قانچىلىك؟',
        'بىر كومىتېت 5 نامزات (Alice, Bob, Charlie, Dave, Eve) گۇرۇپپىسىدىن 2 ئەزانى تاللىشى كېرەك. لېكىن ، Alice بىلەن Bob كومىتېتتا بىللە بولالمايدۇ. كومىتېتنى قانچە خىل ئۇسۇلدا قۇرغىلى بولىدۇ؟',
        'ماشىنا A 8 سائەت ئىشلەيدۇ ، سائىتىگە 100 بىرلىك ئىشلەپچىقىرىدۇ. ماشىنا B 4 سائەت ئىشلەيدۇ ، 1 سائەت دەم ئالىدۇ ، ئاندىن 3 سائەت 120 بىرلىك/سائەت سۈرئەتتە ئىشلەيدۇ. ئومۇمىي مەھسۇلات قانچىلىك؟',
    ],
    'ko': [
        '한 변의 길이가 5cm이고 다른 한 변의 길이가 12cm인 직각삼각형이 있습니다. 세 번째 변의 길이는 얼마입니까?',
        '5명의 후보(Alice, Bob, Charlie, Dave, Eve) 중에서 2명의 위원을 선출해야 합니다. 그러나 Alice와 Bob은 함께 위원이 될 수 없습니다. 위원회를 구성하는 방법은 몇 가지입니까?',
        '기계 A는 8시간 동안 시간당 100개를 생산합니다. 기계 B는 4시간 가동 후 1시간 휴식하고, 그 후 3시간 동안 시간당 120개를 생산합니다. 총 생산량은 얼마입니까?',
    ],
    'es': [
        'Un triángulo tiene lados de longitud 5 cm y 12 cm, con un ángulo recto entre ellos. ¿Cuál es la longitud del tercer lado?',
        'Un comité necesita elegir 2 miembros de un grupo de 5 candidatos (Alice, Bob, Charlie, Dave, Eve). Sin embargo, Alice y Bob no pueden estar juntos en el comité. ¿De cuántas maneras se puede formar el comité?',
        'La máquina A funciona durante 8 horas produciendo 100 unidades/hora. La máquina B funciona durante 4 horas, descansa 1 hora y luego funciona 3 horas a 120 unidades/hora. ¿Cuál es la producción total?',
    ],
    'zh': [
        '一个三角形的两条边长分别为5厘米和12厘米，它们之间夹着一个直角。第三条边的长度是多少？',
        '一个委员会需要从5名候选人（Alice, Bob, Charlie, Dave, Eve）中选出2名成员。但是，Alice和Bob不能同时在委员会中。委员会可以有多少种组成方式？',
        '机器A运行8小时，每小时生产100个单位。机器B运行4小时，休息1小时，然后以每小时120个单位的速度运行3小时。总产量是多少？',
    ],
    'th': [
        'สามเหลี่ยมรูปหนึ่งมีด้านยาว 5 ซม. และ 12 ซม. โดยมีมุมฉากอยู่ระหว่างด้านทั้งสอง ด้านที่สามยาวเท่าไร?',
        'คณะกรรมการต้องการเลือกสมาชิก 2 คนจากกลุ่มผู้สมัคร 5 คน (Alice, Bob, Charlie, Dave, Eve) อย่างไรก็ตาม Alice และ Bob ไม่สามารถอยู่ในคณะกรรมการร่วมกันได้ จะมีกี่วิธีในการจัดตั้งคณะกรรมการ?',
        'เครื่องจักร A ทำงาน 8 ชั่วโมง ผลิตได้ 100 หน่วย/ชั่วโมง เครื่องจักร B ทำงาน 4 ชั่วโมง พัก 1 ชั่วโมง จากนั้นทำงานต่อ 3 ชั่วโมงที่ 120 หน่วย/ชั่วโมง ผลผลิตรวมทั้งหมดคือเท่าไร?',
    ],
    'sw': [
        'Pembetatu ina pande zenye urefu wa sentimita 5 na 12, na kuna pembe mraba kati yake. Je, urefu wa upande wa tatu ni nini?',
        'Kamati inahitaji kuchagua wanachama 2 kutoka kundi la wagombea 5 (Alice, Bob, Charlie, Dave, Eve). Hata hivyo, Alice na Bob hawawezi kuwa kwenye kamati pamoja. Kamati inaweza kuundwa kwa njia ngapi?',
        'Mashine A inafanya kazi kwa masaa 8 ikizalisha vitengo 100/saa. Mashine B inafanya kazi kwa masaa 4, inapumzika saa 1, kisha inafanya kazi kwa masaa 3 kwa vitengo 120/saa. Uzalishaji wote ni upi?',
    ],
    'te': [
        'ఒక త్రిభుజం 5 సెం.మీ మరియు 12 సెం.మీ పొడవు గల భుజాలను కలిగి ఉంది, వాటి మధ్య లంబకోణం ఉంది. మూడవ భుజం పొడవు ఎంత?',
        '5 అభ్యర్థుల (Alice, Bob, Charlie, Dave, Eve) సమూహం నుండి ఒక కమిటీ 2 సభ్యులను ఎన్నుకోవాలి. అయితే, Alice మరియు Bob కలిసి కమిటీలో ఉండలేరు. కమిటీని ఎన్ని విధాలుగా ఏర్పాటు చేయవచ్చు?',
        'మెషిన్ A 8 గంటల పాటు గంటకు 100 యూనిట్లు ఉత్పత్తి చేస్తుంది. మెషిన్ B 4 గంటలు పనిచేసి, 1 గంట విశ్రాంతి తీసుకుని, తర్వాత 3 గంటల పాటు గంటకు 120 యూనిట్ల చొప్పున పనిచేస్తుంది. మొత్తం ఉత్పత్తి ఎంత?',
    ],
}

# === Skeleton few-shot ANSWERS, keyed by skeleton language (ell_s) ===
SKELETON_ANSWER_FEWSHOTS = {
    "en": [
        "**Problem Structure**\n- Geometric figure: right triangle with three sides\n- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n- Unknown: third side (hypotenuse) connecting the two legs\n- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n**Key Concepts / Tools**\n- Pythagorean theorem\n- Right triangle geometry",
        "**Problem Structure**\n- Discrete selection problem: choosing k=2 elements from n=5 candidates\n- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n- Unknown: number of valid selection configurations\n- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n\n**Key Concepts / Tools**\n- Combinatorial counting (combinations)\n- Constraint satisfaction\n- Complementary counting",
        "**Problem Structure**\n- Two independent production processes with different temporal patterns\n- Variables: production rates, time durations, total output\n- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour)\n\n**Key Concepts / Tools**\n- Rate-time-quantity relationships\n- Piecewise continuous processes\n- Linear aggregation",
    ],
    "ko": [
        "**문제 구조**\n- 기하학적 도형: 세 변을 가진 직각삼각형\n- 알려진 양: 두 수직 변 (5 cm, 12 cm), 직각 조건\n- 미지수: 두 다리를 연결하는 세 번째 변 (빗변)\n- 관계: 변의 길이는 직각 속성에 의해 강제되는 거리 관계를 만족\n\n**핵심 개념 / 도구**\n- 피타고라스 정리\n- 직각삼각형 기하학",
        "**문제 구조**\n- 이산 선택 문제: n=5명의 후보에서 k=2명의 요소 선택\n- 알려진 것: 총 후보 수 (5), 위원회 크기 (2), 쌍별 배제 제약\n- 미지수: 유효한 선택 구성의 수\n- 제약: 쌍별 배제 (Alice와 Bob은 함께 있을 수 없음)\n\n**핵심 개념 / 도구**\n- 조합 계산 (조합)\n- 제약 충족\n- 여집합 계산",
        "**문제 구조**\n- 서로 다른 시간 패턴을 가진 두 개의 독립적인 생산 프로세스\n- 변수: 생산율, 시간 지속, 총 출력\n- 기계 A: 단일 연속 활성 구간 (8시간 @ 100 단위/시간)\n- 기계 B: 두 개의 분리된 활성 구간 (4시간 + 3시간 @ 120 단위/시간)\n\n**핵심 개념 / 도구**\n- 속도-시간-수량 관계\n- 구간별 연속 프로세스\n- 선형 집계",
    ],
    "es": [
        "**Estructura del Problema**\n- Figura geométrica: triángulo rectángulo con tres lados\n- Cantidades conocidas: dos lados perpendiculares (5 cm, 12 cm), restricción de ángulo recto\n- Incógnita: tercer lado (hipotenusa) que conecta los dos catetos\n- Relación: las longitudes de los lados satisfacen la relación de distancia impuesta por la propiedad del ángulo recto\n\n**Conceptos Clave / Herramientas**\n- Teorema de Pitágoras\n- Geometría del triángulo rectángulo",
        "**Estructura del Problema**\n- Problema de selección discreta: elegir k=2 elementos de n=5 candidatos\n- Conocido: total de candidatos (5), tamaño del comité (2), restricción de exclusión por pares\n- Incógnita: número de configuraciones de selección válidas\n- Restricción: exclusión por pares (Alice y Bob no pueden coincidir)\n\n**Conceptos Clave / Herramientas**\n- Conteo combinatorio (combinaciones)\n- Satisfacción de restricciones\n- Conteo complementario",
        "**Estructura del Problema**\n- Dos procesos de producción independientes con diferentes patrones temporales\n- Variables: tasas de producción, duraciones de tiempo, producción total\n- Máquina A: intervalo activo continuo único (8 horas @ 100 unidades/hora)\n- Máquina B: dos intervalos activos disjuntos (4h + 3h @ 120 unidades/hora)\n\n**Conceptos Clave / Herramientas**\n- Relaciones tasa-tiempo-cantidad\n- Procesos continuos por tramos\n- Agregación lineal",
    ],
    "zh": [
        "**问题结构**\n- 几何图形：具有三条边的直角三角形\n- 已知量：两条垂直边（5厘米，12厘米），直角约束\n- 未知量：连接两条直角边的第三边（斜边）\n- 关系：边长满足由直角属性强制执行的距离关系\n\n**关键概念/工具**\n- 勾股定理\n- 直角三角形几何",
        "**问题结构**\n- 离散选择问题：从n=5个候选人中选择k=2个元素\n- 已知：候选人总数（5），委员会规模（2），成对排斥约束\n- 未知量：有效选择配置的数量\n- 约束：成对排斥（Alice和Bob不能同时出现）\n\n**关键概念/工具**\n- 组合计数（组合）\n- 约束满足\n- 补集计数",
        "**问题结构**\n- 两个具有不同时间模式的独立生产过程\n- 变量：生产率，时间持续，总产量\n- 机器A：单个连续活动区间（8小时 @ 100单位/小时）\n- 机器B：两个不相交的活动区间（4小时 + 3小时 @ 120单位/小时）\n\n**关键概念/工具**\n- 速率-时间-数量关系\n- 分段连续过程\n- 线性聚合",
    ],
    "th": [
        "**โครงสร้างปัญหา**\n- รูปทรงเรขาคณิต: สามเหลี่ยมมุมฉากที่มีด้านสามด้าน\n- ปริมาณที่ทราบ: ด้านตั้งฉากสองด้าน (5 ซม., 12 ซม.), ข้อจำกัดมุมฉาก\n- ไม่ทราบ: ด้านที่สาม (ด้านตรงข้ามมุมฉาก) ที่เชื่อมต่อขาทั้งสอง\n- ความสัมพันธ์: ความยาวด้านเป็นไปตามความสัมพันธ์ระยะทางที่บังคับโดยคุณสมบัติมุมฉาก\n\n**แนวคิดหลัก / เครื่องมือ**\n- ทฤษฎีบทพีทาโกรัส\n- เรขาคณิตสามเหลี่ยมมุมฉาก",
        "**โครงสร้างปัญหา**\n- ปัญหาการเลือกแบบไม่ต่อเนื่อง: เลือก k=2 องค์ประกอบจาก n=5 ผู้สมัคร\n- ทราบ: ผู้สมัครทั้งหมด (5), ขนาดคณะกรรมการ (2), ข้อจำกัดการยกเว้นคู่\n- ไม่ทราบ: จำนวนการกำหนดค่าการเลือกที่ถูกต้อง\n- ข้อจำกัด: การยกเว้นคู่\n\n**แนวคิดหลัก / เครื่องมือ**\n- การนับแบบผสม (การรวมกัน)\n- การตอบสนองข้อจำกัด\n- การนับเสริม",
        "**โครงสร้างปัญหา**\n- กระบวนการผลิตอิสระสองกระบวนการที่มีรูปแบบเวลาต่างกัน\n- ตัวแปร: อัตราการผลิต, ระยะเวลา, ผลผลิตรวม\n- เครื่องจักร A: ช่วงเวลาทำงานต่อเนื่องเดียว (8 ชั่วโมง @ 100 หน่วย/ชั่วโมง)\n- เครื่องจักร B: ช่วงเวลาทำงานแยกกันสองช่วง (4 ชม. + 3 ชม. @ 120 หน่วย/ชั่วโมง)\n\n**แนวคิดหลัก / เครื่องมือ**\n- ความสัมพันธ์อัตรา-เวลา-ปริมาณ\n- กระบวนการต่อเนื่องแบบชิ้นส่วน\n- การรวมแบบเชิงเส้น",
    ],
    "ru": [
        "**Структура задачи**\n- Геометрическая фигура: прямоугольный треугольник с тремя сторонами\n- Известные величины: две перпендикулярные стороны (5 см, 12 см), ограничение прямого угла\n- Неизвестное: третья сторона (гипотенуза), соединяющая два катета\n- Отношение: длины сторон удовлетворяют соотношению расстояний, обусловленному свойством прямого угла\n\n**Ключевые концепции / Инструменты**\n- Теорема Пифагора\n- Геометрия прямоугольного треугольника",
        "**Структура задачи**\n- Дискретная задача выбора: выбор k=2 элементов из n=5 кандидатов\n- Известно: общее количество кандидатов (5), размер комитета (2), ограничение попарного исключения\n- Неизвестное: количество допустимых конфигураций выбора\n- Ограничение: попарное исключение (Алиса и Боб не могут быть вместе)\n\n**Ключевые концепции / Инструменты**\n- Комбинаторный подсчёт (комбинации)\n- Удовлетворение ограничений\n- Дополнительный подсчёт",
        "**Структура задачи**\n- Два независимых производственных процесса с разными временными паттернами\n- Переменные: производственные мощности, временные интервалы, общий выпуск\n- Машина A: единый непрерывный активный интервал (8 часов @ 100 единиц/час)\n- Машина B: два непересекающихся активных интервала (4ч + 3ч @ 120 единиц/час)\n\n**Ключевые концепции / Инструменты**\n- Соотношения скорость-время-количество\n- Кусочно-непрерывные процессы\n- Линейная агрегация",
    ],
}

# === Translation few-shots (5-shot), keyed by '<src>_to_en' ===
TRANSLATION_FEWSHOTS = {
    "ko_to_en": [
        {"src": "밧줄을 두 조각으로 잘랐습니다. 한 조각은 4미터이고, 다른 한 조각은 그 길이의 절반입니다. 전체 길이는 얼마입니까?",
         "tgt": "A rope is cut into two pieces: one is 4 meters long, and the other is half as long. What is the total length?"},
        {"src": "반지름이 3인 원 A와 반지름이 5인 원 B가 있습니다. 두 원의 중심 사이의 거리는 10입니다. 두 원의 공통 내접선의 길이는 얼마입니까?",
         "tgt": "Circle A has a radius of 3, and Circle B has a radius of 5. The distance between their centers is 10. What is the length of the common internal tangent segment?"},
        {"src": "5권의 서로 다른 책을 책장에 배열할 때, 특정 책 A와 B가 서로 이웃하지 않도록 배열하는 방법의 수는 몇 가지입니까?",
         "tgt": "In how many ways can 5 different books be arranged on a shelf such that two specific books, A and B, are never placed next to each other?"},
        {"src": "어떤 수에 5를 더한 후 3을 곱하면 24가 됩니다. 그 수는 무엇입니까?",
         "tgt": "If you add 5 to a number and then multiply by 3, you get 24. What is the number?"},
        {"src": "세 변의 길이가 3, 4, 5인 직각삼각형의 넓이를 구하시오.",
         "tgt": "Find the area of a right triangle with sides of length 3, 4, and 5."},
    ],
    "sw_to_en": [
        {"src": "Kamba imekatwa vipande viwili: kipande kimoja kina urefu wa mita 4, na kingine kina urefu wa nusu ya kipande hicho. Je, jumla ya urefu wa kamba ni kiasi gani?",
         "tgt": "A rope is cut into two pieces: one is 4 meters long, and the other is half as long. What is the total length?"},
        {"src": "Duara A lina radiusi ya 3, na Duara B lina radiusi ya 5. Umbali kati ya vituo vyao ni 10. Je, urefu wa mstari wa kugusana wa ndani ni kiasi gani?",
         "tgt": "Circle A has a radius of 3, and Circle B has a radius of 5. The distance between their centers is 10. What is the length of the common internal tangent segment?"},
        {"src": "Vitabu 5 tofauti vinaweza kupangwa kwa njia ngapi kwenye rafu ili vitabu maalum A na B visiwe kamwe vimewekwa viwili kwa pamoja?",
         "tgt": "In how many ways can 5 different books be arranged on a shelf such that two specific books, A and B, are never placed next to each other?"},
        {"src": "Ukijumlisha 5 kwa nambari fulani kisha kuzidisha kwa 3, unapata 24. Nambari hiyo ni ipi?",
         "tgt": "If you add 5 to a number and then multiply by 3, you get 24. What is the number?"},
        {"src": "Tafuta eneo la pembetatu ya pembe moja-sifuri yenye pande za urefu 3, 4, na 5.",
         "tgt": "Find the area of a right triangle with sides of length 3, 4, and 5."},
    ],
    "es_to_en": [
        {"src": "Una cuerda se corta en dos pedazos: uno mide 4 metros y el otro mide la mitad. ¿Cuál es la longitud total?",
         "tgt": "A rope is cut into two pieces: one is 4 meters long, and the other is half as long. What is the total length?"},
        {"src": "El círculo A tiene un radio de 3 y el círculo B tiene un radio de 5. La distancia entre sus centros es 10. ¿Cuál es la longitud del segmento tangente interno común?",
         "tgt": "Circle A has a radius of 3, and Circle B has a radius of 5. The distance between their centers is 10. What is the length of the common internal tangent segment?"},
        {"src": "¿De cuántas maneras se pueden ordenar 5 libros diferentes en un estante de modo que dos libros específicos, A y B, nunca estén uno al lado del otro?",
         "tgt": "In how many ways can 5 different books be arranged on a shelf such that two specific books, A and B, are never placed next to each other?"},
        {"src": "Si sumas 5 a un número y luego multiplicas por 3, obtienes 24. ¿Cuál es el número?",
         "tgt": "If you add 5 to a number and then multiply by 3, you get 24. What is the number?"},
        {"src": "Encuentra el área de un triángulo rectángulo con lados de longitud 3, 4 y 5.",
         "tgt": "Find the area of a right triangle with sides of length 3, 4, and 5."},
    ],
    "zh_to_en": [
        {"src": "一根绳子被切成两段：一段长4米，另一段是它的一半。总长度是多少？",
         "tgt": "A rope is cut into two pieces: one is 4 meters long, and the other is half as long. What is the total length?"},
        {"src": "圆A的半径为3，圆B的半径为5。它们的圆心之间的距离是10。公共内切线段的长度是多少？",
         "tgt": "Circle A has a radius of 3, and Circle B has a radius of 5. The distance between their centers is 10. What is the length of the common internal tangent segment?"},
        {"src": "5本不同的书可以以多少种方式排列在书架上，使得特定的两本书A和B永远不会相邻？",
         "tgt": "In how many ways can 5 different books be arranged on a shelf such that two specific books, A and B, are never placed next to each other?"},
        {"src": "如果你将一个数加上5然后乘以3，得到24。这个数是多少？",
         "tgt": "If you add 5 to a number and then multiply by 3, you get 24. What is the number?"},
        {"src": "求边长为3、4、5的直角三角形的面积。",
         "tgt": "Find the area of a right triangle with sides of length 3, 4, and 5."},
    ],
    "th_to_en": [
        {"src": "เชือกถูกตัดเป็นสองส่วน: ส่วนหนึ่งยาว 4 เมตร และอีกส่วนยาวครึ่งหนึ่ง ความยาวทั้งหมดเป็นเท่าไร?",
         "tgt": "A rope is cut into two pieces: one is 4 meters long, and the other is half as long. What is the total length?"},
        {"src": "วงกลม A มีรัศมี 3 และวงกลม B มีรัศมี 5 ระยะห่างระหว่างจุดศูนย์กลางคือ 10 ความยาวของเส้นสัมผัสภายในร่วมเป็นเท่าไร?",
         "tgt": "Circle A has a radius of 3, and Circle B has a radius of 5. The distance between their centers is 10. What is the length of the common internal tangent segment?"},
        {"src": "หนังสือ 5 เล่มที่แตกต่างกันสามารถจัดเรียงบนชั้นได้กี่วิธี โดยที่หนังสือ A และ B ไม่อยู่ติดกัน?",
         "tgt": "In how many ways can 5 different books be arranged on a shelf such that two specific books, A and B, are never placed next to each other?"},
        {"src": "ถ้าคุณบวก 5 กับตัวเลขหนึ่งแล้วคูณด้วย 3 คุณจะได้ 24 ตัวเลขนั้นคืออะไร?",
         "tgt": "If you add 5 to a number and then multiply by 3, you get 24. What is the number?"},
        {"src": "หาพื้นที่ของสามเหลี่ยมมุมฉากที่มีด้านยาว 3, 4 และ 5",
         "tgt": "Find the area of a right triangle with sides of length 3, 4, and 5."},
    ],
    "te_to_en": [
        {"src": "ఒక తాడును రెండు ముక్కలుగా కత్తిరించారు: ఒకటి 4 మీటర్లు పొడవు, మరొకటి దాని సగం. మొత్తం పొడవు ఎంత?",
         "tgt": "A rope is cut into two pieces: one is 4 meters long, and the other is half as long. What is the total length?"},
        {"src": "వృత్తం A యొక్క వ్యాసార్धం 3, మరియు వృత్తం B యొక్క వ్యాసార్థం 5. వాటి కేంద్రాల మధ్య దూరం 10. ఉమ్మడి అంతర్గత స్పర్శరేఖ భాగం యొక్క పొడవు ఎంత?",
         "tgt": "Circle A has a radius of 3, and Circle B has a radius of 5. The distance between their centers is 10. What is the length of the common internal tangent segment?"},
        {"src": "5 వేర్వేరు పుస్తకాలను షెల్ఫ్‌లో ఎన్ని విధాలుగా అమర్చవచ్చు, A మరియు B అనే రెండు నిర్దిష్ట పుస్తకాలు ఎప్పుడూ ప్రక్కన ఉండకూడదు?",
         "tgt": "In how many ways can 5 different books be arranged on a shelf such that two specific books, A and B, are never placed next to each other?"},
        {"src": "మీరు ఒక సంఖ్యకు 5 కలిపి 3 తో గుణిస్తే 24 వస్తుంది. ఆ సంఖ్య ఏమిటి?",
         "tgt": "If you add 5 to a number and then multiply by 3, you get 24. What is the number?"},
        {"src": "3, 4, 5 భుజాలు కలిగిన లంబ త్రిభుజం యొక్క వైశాల్యం కనుగొనండి.",
         "tgt": "Find the area of a right triangle with sides of length 3, 4, and 5."},
    ],
}

# === Solver start-triggers per reasoning/answer language (ell_a) ===
SOLVER_TRIGGERS = {
        # 인도 언어들
        "ta": "சரி,\n",  # Tamil (Sari - Okay)
        "kn": "ಸರಿ,\n",  # Kannada (Sari - Okay)
        "si": "හොඳයි,\n",  # Sinhala (Hondai - Good)
        "gu": "સારું,\n",  # Gujarati (Saru - Good)
        "ne": "हुन्छ,\n",  # Nepali (Huncha - Okay/Will do)
        "ml": "ശരി,\n",  # Malayalam (Sari - Okay)
        "mr": "ठीक,\n",  # Marathi (Thik - Okay)
        "or": "ଠିକ୍,\n",  # Odia (Thik - Okay)
        "pa": "ਠੀਕ,\n",  # Punjabi (Thik - Okay)
        # 동남아시아
        "my": "ကောင်းပြီ,\n",  # Burmese (Kaung byi - Good)
        "km": "ល្អ,\n",  # Khmer (L'or - Good)
        "lo": "ໂດຍ,\n",  # Lao (Doy - Polite Yes/Ack) *수정됨*
        "jv": "Inggih,\n",  # Javanese (Inggih - Polite Yes) *수정됨*
        "su": "Leres,\n",  # Sundanese (Leres - Right/Polite)
        "ceb": "Sige,\n",  # Cebuano (Sige - Go ahead/Okay)
        # 중앙아시아 / 튀르크어
        "uz": "Yaxshi,\n",  # Uzbek (Good)
        "ky": "Жакшы,\n",  # Kyrgyz (Good)
        "kk": "Жақсы,\n",  # Kazakh (Good)
        "tg": "Хуб,\n",  # Tajik (Good)
        "ug": "ياخشى,\n",  # Uyghur (Yaxshi - Good)
        # 캅카스 / 아르메니아
        "ka": "კარგი,\n",  # Georgian (Kargi - Good)
        "hy": "Լավ,\n",  # Armenian (Lav - Good) *수정됨*
        # 아프리카
        "am": "እሺ,\n",  # Amharic (Eshi - Okay)
        "yo": "O da,\n",  # Yoruba (It is good)
        "so": "Hagaag,\n",  # Somali (Fine/Okay)
        "mg": "Eny,\n",  # Malagasy (Yes/Okay)
        # 유럽
        "eu": "Ados,\n",  # Basque (Agreed)
        "mt": "Tajjeb,\n",  # Maltese (Good)
        # 중동
        "ku": "Baş e,\n",  # Kurdish (It is good)
        "ps": "ښه,\n",  # Pashto (Good)
        "sd": "ٺيڪ,\n",  # Sindhi (Okay)
        # 아메리카 원주민
        "gn": "Oĩma,\n",  # Guarani (Ready/Okay)
        "qu": "Allin,\n",  # Quechua (Good)
        # 기타
        "mn": "За,\n",  # Mongolian (Okay/Well)
    }

# === CoT start-triggers (non-reasoning models) ===
COT_TRIGGERS = {
        "ko": "좋습니다,", "sw": "Sawa,", "es": "Bien,",
        "zh": "好的，", "th": "ดี", "te": "సరే,", "en": ""
    }

# === CoT start-triggers (reasoning models, inside <think>) ===
COT_THINK_TRIGGERS = {
        "ko": "좋아,", "sw": "Sawa,", "es": "Bien,",
        "zh": "好的，", "th": "ดี", "te": "సరే,", "en": ""
    }


def get_lang_name(code: str) -> str:
    """Return the English name of a language code (fallback: the code itself)."""
    return LANGUAGE_NAMES.get(code, code)

