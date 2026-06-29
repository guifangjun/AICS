"""
微信回复构建器：将回复内容包装为微信 XML 格式。

支持的消息类型：
  - 文本消息（text）
  - 图文消息（news）
  - 客服消息（通过微信客服消息 API 主动推送）

微信 XML 回复格式：
  <xml>
    <ToUserName><![CDATA[openid]]></ToUserName>
    <FromUserName><![CDATA[公众号]]></FromUserName>
    <CreateTime>时间戳</CreateTime>
    <MsgType><![CDATA[text]]></MsgType>
    <Content><![CDATA[回复内容]]></Content>
  </xml>
"""

import time
from typing import List, Dict


class ReplyBuilder:
    """将内部回复对象序列化为微信 XML 消息。"""

    @staticmethod
    def _build_xml(from_user: str, to_user: str, body_elements: str) -> str:
        """拼接标准 XML 外壳。"""
        return f"""<xml>
<ToUserName><![CDATA[{to_user}]]></ToUserName>
<FromUserName><![CDATA[{from_user}]]></FromUserName>
<CreateTime>{int(time.time())}</CreateTime>
{body_elements}
</xml>"""

    def build_text_reply(self, from_user: str, to_user: str, content: str) -> str:
        """
        构建文本消息回复。

        参数:
            from_user: 公众号（回复方）OpenID
            to_user: 用户（接收方）OpenID
            content: 回复的文本内容
        """
        return self._build_xml(
            from_user, to_user,
            f"<MsgType><![CDATA[text]]></MsgType>\n<Content><![CDATA[{content}]]></Content>",
        )

    def build_news_reply(
        self, from_user: str, to_user: str, articles: List[Dict[str, str]]
    ) -> str:
        """
        构建图文消息回复。

        参数:
            articles: 图文列表，每项含 Title, Description, PicUrl, Url
        """
        article_count = len(articles)
        articles_xml = []
        for art in articles:
            articles_xml.append(f"""<item>
<Title><![CDATA[{art.get('Title', '')}]]></Title>
<Description><![CDATA[{art.get('Description', '')}]]></Description>
<PicUrl><![CDATA[{art.get('PicUrl', '')}]]></PicUrl>
<Url><![CDATA[{art.get('Url', '')}]]></Url>
</item>""")

        return self._build_xml(
            from_user, to_user,
            f"<MsgType><![CDATA[news]]></MsgType>\n<ArticleCount>{article_count}</ArticleCount>\n<Articles>\n{''.join(articles_xml)}\n</Articles>",
        )

    def build_default_reply(self, from_user: str, to_user: str) -> str:
        """默认兜底回复（非文本消息时使用）。"""
        return self.build_text_reply(
            from_user, to_user,
            "收到啦～目前我只支持文字交流哦，有什么问题直接打字告诉我吧！",
        )
