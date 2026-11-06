from pydantic import BaseModel
from instructor import patch
from openai import AsyncOpenAI
from langdetect import detect

docs = map(
    lambda x: x.strip(),
    """
Լեզվական մոդելները վերջին տարիներին դարձել են ավելի հարուստ և կատարյալ, հնարավորություն ընձեռելով ստեղծել սահուն և բնական տեքստեր, ինչպես նաև գերազանց արդյունքներ ցուցաբերել մեքենայական թարգմանության, հարցերի պատասխանման և ստեղծագործ տեքստերի ստեղծման նման տարբեր առաջադրանքներում։ Այս մոդելները մշակվում են հսկայական տեքստային տվյալների հիման վրա և կարող են բռնել բնական լեզվի կառուցվածքն ու նրբությունները՝ հեղափոխություն առաջացնելով համակարգիչների և մարդկանց միջև հաղորդակցության ոլորտում։

---

Mga modelo ng wika ay naging mas sopistikado sa nagdaang mga taon, na nagbibigay-daan sa pagbuo ng mga natural at madaling basahing teksto, at nagpapakita ng mahusay na pagganap sa iba't ibang gawain tulad ng awtomatikong pagsasalin, pagsagot sa mga tanong, at pagbuo ng malikhain na teksto. Ang mga modelo na ito ay sinanay sa napakalaking mga dataset ng teksto at kayang hulihin ang istruktura at mga nuances ng natural na wika. Ang mga pagpapabuti sa mga modelo ng wika ay maaaring magdulot ng rebolusyon sa komunikasyon sa pagitan ng mga computer at tao, at inaasahan ang higit pang pag-unlad sa hinaharap.

---

Ngaahi motuʻa lea kuo nau hoko ʻo fakaʻofoʻofa ange ʻi he ngaahi taʻu fakamuimui ni, ʻo fakafaingofuaʻi e fakatupu ʻo e ngaahi konga tohi ʻoku lelei mo fakanatula pea ʻoku nau fakahaaʻi ʻa e ngaahi ola lelei ʻi he ngaahi ngāue kehekehe ʻo hangē ko e liliu fakaʻētita, tali fehuʻi, mo e fakatupu ʻo e konga tohi fakaʻatamai. Ko e ako ʻa e ngaahi motuʻa ni ʻi he ngaahi seti ʻo e fakamatala tohi lahi pea ʻoku nau malava ʻo puke ʻa e fakafuofua mo e ngaahi meʻa iiki ʻo e lea fakanatula. ʻE lava ke fakatupu ʻe he ngaahi fakaleleiʻi ki he ngaahi motuʻa lea ha liliu lahi ʻi he fetu'utaki ʻi he vahaʻa ʻo e ngaahi komipiuta mo e kakai, pea ʻoku ʻamanaki ʻe toe fakalakalaka ange ia ʻi he kahaʻu.

---

Dil modelleri son yıllarda daha da gelişti, akıcı ve doğal metinler üretmeyi mümkün kılıyor ve makine çevirisi, soru cevaplama ve yaratıcı metin oluşturma gibi çeşitli görevlerde mükemmel performans gösteriyor. Bu modeller, devasa metin veri setlerinde eğitilir ve doğal dilin yapısını ve nüanslarını yakalayabilir. Dil modellerindeki iyileştirmeler, bilgisayarlar ve insanlar arasındaki iletişimde devrim yaratabilir ve gelecekte daha da ilerleme bekleniyor.

---

Mô hình ngôn ngữ đã trở nên tinh vi hơn trong những năm gần đây, cho phép tạo ra các văn bản trôi chảy và tự nhiên, đồng thời thể hiện hiệu suất xuất sắc trong các nhiệm vụ khác nhau như dịch máy, trả lời câu hỏi và tạo văn bản sáng tạo. Các mô hình này được huấn luyện trên các tập dữ liệu văn bản khổng lồ và có thể nắm bắt cấu trúc và sắc thái của ngôn ngữ tự nhiên. Những cải tiến trong mô hình ngôn ngữ có thể mang lại cuộc cách mạng trong giao tiếp giữa máy tính và con người, và người ta kỳ vọng sẽ có những tiến bộ hơn nữa trong tương lai.

---

Les modèles de langage sont devenus de plus en plus sophistiqués ces dernières années, permettant de générer des textes fluides et naturels, et de performer dans une variété de tâches telles que la traduction automatique, la réponse aux questions et la génération de texte créatif. Entraînés sur d'immenses ensembles de données textuelles, ces modèles sont capables de capturer la structure et les nuances du langage naturel, ouvrant la voie à une révolution dans la communication entre les ordinateurs et les humains.

---

近年来,语言模型变得越来越复杂,能够生成流畅自然的文本,并在机器翻译、问答和创意文本生成等各种任务中表现出色。这些模型在海量文本数据集上训练,可以捕捉自然语言的结构和细微差别。语言模型的改进有望彻底改变计算机和人类之间的交流方式,未来有望实现更大的突破。

---

In den letzten Jahren sind Sprachmodelle immer ausgefeilter geworden und können flüssige, natürlich klingende Texte generieren und in verschiedenen Aufgaben wie maschineller Übersetzung, Beantwortung von Fragen und Generierung kreativer Texte hervorragende Leistungen erbringen. Diese Modelle werden auf riesigen Textdatensätzen trainiert und können die Struktur und Nuancen natürlicher Sprache erfassen, was zu einer Revolution in der Kommunikation zwischen Computern und Menschen führen könnte.

---

पिछले कुछ वर्षों में भाषा मॉडल बहुत अधिक परिष्कृत हो गए हैं, जो प्राकृतिक और प्रवाहमय पाठ उत्पन्न कर सकते हैं, और मशीन अनुवाद, प्रश्नोत्तर, और रचनात्मक पाठ उत्पादन जैसे विभिन्न कार्यों में उत्कृष्ट प्रदर्शन कर सकते हैं। ये मॉडल विशाल पाठ डेटासेट पर प्रशिक्षित होते हैं और प्राकृतिक भाषा की संरचना और बारीकियों को समझ सकते हैं। भाषा मॉडल में सुधार कंप्यूटर और मानव के बीच संवाद में क्रांति ला सकता है, और भविष्य में और प्रगति की उम्मीद है।

---

近年、言語モデルは非常に洗練され、自然で流暢なテキストを生成できるようになり、機械翻訳、質問応答、クリエイティブなテキスト生成など、様々なタスクで優れたパフォーマンスを発揮しています。これらのモデルは膨大なテキストデータセットで学習され、自然言語の構造とニュアンスを捉えることができます。言語モデルの改善により、コンピューターと人間のコミュニケーションに革命が起こる可能性があり、将来のさらなる進歩が期待されています。
""".split("---"),
)

# Patch the OpenAI client to enable response_model
client = patch(AsyncOpenAI())


class GeneratedSummary(BaseModel):
    summary: str


async def summarize_text(text: str):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=GeneratedSummary,
        messages=[
            {
                "role": "system",
                "content": "Generate a concise summary in the language of the article. ",
            },
            {
                "role": "user",
                "content": f"Summarize the following text in a concise way:\n{text}",
            },
        ],
    )  # type: ignore
    return response.summary, text


if __name__ == "__main__":
    import asyncio

    async def main():
        results = await asyncio.gather(*[summarize_text(doc) for doc in docs])
        for summary, doc in results:
            source_lang = detect(doc)
            target_lang = detect(summary)
            print(
                f"Source: {source_lang}, Summary: {target_lang}, Match: {source_lang == target_lang}"
            )

    asyncio.run(main())
    """
    Source: et, Summary: en, Match: False
    Source: tl, Summary: tl, Match: True
    Source: sw, Summary: en, Match: False
    Source: tr, Summary: tr, Match: True
    Source: vi, Summary: en, Match: False
    Source: fr, Summary: fr, Match: True
    Source: zh-cn, Summary: en, Match: False
    Source: de, Summary: de, Match: True
    Source: hi, Summary: en, Match: False
    Source: ja, Summary: en, Match: False
    """
