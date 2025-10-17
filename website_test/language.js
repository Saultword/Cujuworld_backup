// language.js - 多语言支持
const translations = {
    'en': {
        // 通用文本
        'site-title': 'CujuWorld',
        'site-by': 'by cuju.world',
        'menu': 'Menu',
        'homepage': 'Homepage',
        'style-recognition': 'Style Recognition',
        'create-postcard': 'Create your postcard!',
        'recent-works': 'Recent Works',
        'more': 'More',
        'get-in-touch': 'Get in touch',
        'contact-us': 'Contact us',
        'search-placeholder': 'Search',
        
        // index.html 特定文本
        'make-soccer-fun': 'Make soccer more fun',
        'transform-moments': 'Transform your sports moments into iconic masterpieces with StarStyle AI!',
        'learn-more': 'Learn More',
        
        // generic.html 特定文本
        'which-player': 'Which soccer player are you most like?',
        'drag-drop': 'Drag & Drop your video here',
        'select-preset': 'Select preset test video:',
        'please-select': '-- Please select --',
        'load-video': 'Load video',
        'analysis': 'Analysis',
        'analyzing': 'Analyzing...',
        'result': 'Result',
        'highlight': 'Highlight!',
        'history': 'History',
        'no-history': 'No history yet',
        'segmented-analysis': 'Segmented Analysis',
        'upload-find-superstar': 'Upload the video let us find who is the superstar you would be!',
        
        // elements.html 特定文本
        'generate-yourself': 'Generate one for yourself!',
        'generate-unique-photo': 'Generate the unique photo of yourself!',
        'think-ronaldo': 'Think about being Cristiano Ronaldo.',
        'postcard-generation': 'Postcard generation',
        'lets-begin': 'Let\'s begin',
        'origin': 'Origin',
        'after': 'After',
        'generate': 'Generate',
        'generating': 'Generating...',
        'select-frame': 'Please select a frame',
        'generate-success': 'Generated successfully!',
        'generate-failed': 'Generation failed: ',
        
        // 操作指引文本
        'operation-guide': 'Operation Guide',
        'guide-step1': 'Upload Video',
        'guide-step1-desc': 'Click "Select Video File" button or drag and drop video file to the designated area',
        'guide-step2': 'Preview Video',
        'guide-step2-desc': 'Confirm the uploaded video is correct, use player controls to adjust playback',
        'guide-step3': 'Analyze Movements',
        'guide-step3-desc': 'Click "Analysis" button to start analyzing your football playing style',
        'guide-step4': 'View Results',
        'guide-step4-desc': 'Check system analysis results to see which football star\'s style matches yours',
        'guide-step5': 'Generate Poster',
        'guide-step5-desc': 'Go to "Create your postcard!" page to generate personalized poster',
        'poster-creation-guide': 'Poster Creation Guide',
        'poster-step1': 'Select Key Frame',
        'poster-step1-desc': 'Choose your favorite sports moment key frame from history',
        'poster-step2': 'Generate Poster',
        'poster-step2-desc': 'Click "Generate" button to create your exclusive star-style poster',
        'poster-step3': 'Preview Effect',
        'poster-step3-desc': 'View the AI-generated result to ensure you\'re satisfied with the final poster'
    },
    'zh': {
        // 通用文本
        'site-title': '蹴鞠世界',
        'site-by': '由 cuju.world 提供',
        'menu': '選單',
        'homepage': '首頁',
        'style-recognition': '風格識別',
        'create-postcard': '創建您的明信片！',
        'recent-works': '最近作品',
        'more': '更多',
        'get-in-touch': '聯繫我們',
        'contact-us': '聯繫我們',
        'search-placeholder': '搜索',
        
        // index.html 特定文本
        'make-soccer-fun': '讓足球更有趣',
        'transform-moments': '使用StarStyle AI將您的運動時刻轉變為標誌性傑作！',
        'learn-more': '了解更多',
        
        // generic.html 特定文本
        'which-player': '您最像哪位足球運動員？',
        'drag-drop': '拖放您的視頻到此處',
        'select-preset': '選擇預設測試視頻：',
        'please-select': '-- 請選擇 --',
        'load-video': '加載視頻',
        'analysis': '分析',
        'analyzing': '分析中...',
        'result': '結果',
        'highlight': '精彩時刻！',
        'history': '歷史記錄',
        'no-history': '暫無歷史記錄',
        'segmented-analysis': '分段分析',
        'upload-find-superstar': '上傳視頻，讓我們找到您最像的巨星！',
        
        // elements.html 特定文本
        'generate-yourself': '為自己生成一個！',
        'generate-unique-photo': '生成您獨一無二的照片！',
        'think-ronaldo': '想像自己是Cristiano Ronaldo。',
        'postcard-generation': '明信片生成',
        'lets-begin': '開始吧',
        'origin': '原始',
        'after': '之後',
        'generate': '生成',
        'generating': '生成中...',
        'select-frame': '請選擇一個畫面',
        'generate-success': '生成成功！',
        'generate-failed': '生成失敗：',
        
        // 操作指引文本
        'operation-guide': '操作指南',
        'guide-step1': '上傳視頻',
        'guide-step1-desc': '點擊"選擇視頻文件"按鈕或拖放視頻文件到指定區域',
        'guide-step2': '預覽視頻',
        'guide-step2-desc': '確認上傳的視頻正確，使用播放器控制調整播放',
        'guide-step3': '分析動作',
        'guide-step3-desc': '點擊"分析"按鈕開始分析您的足球風格',
        'guide-step4': '查看結果',
        'guide-step4-desc': '查看系統分析結果，了解您最像哪位足球明星',
        'guide-step5': '生成海報',
        'guide-step5-desc': '前往"創建您的明信片！"頁面生成個性化海報',
        'poster-creation-guide': '海報創建指南',
        'poster-step1': '選擇關鍵畫面',
        'poster-step1-desc': '從歷史記錄中選擇您最喜歡的運動時刻關鍵畫面',
        'poster-step2': '生成海報',
        'poster-step2-desc': '點擊"生成"按鈕創建專屬明星風格海報',
        'poster-step3': '預覽效果',
        'poster-step3-desc': '查看AI生成的結果，確保您對最終海報滿意'
    }
};

// 当前语言
let currentLanguage = 'en';

// 初始化语言
function initLanguage() {
    // 检查本地存储中的语言偏好
    const savedLanguage = localStorage.getItem('preferredLanguage');
    if (savedLanguage) {
        currentLanguage = savedLanguage;
    } else {
        // 根据浏览器语言设置默认语言
        const browserLang = navigator.language || navigator.userLanguage;
        if (browserLang.startsWith('zh')) {
            currentLanguage = 'zh';
        }
    }
    
    // 更新按钮状态
    updateLanguageButtons();
    
    // 应用语言
    applyLanguage();
}

// 更新语言按钮状态
function updateLanguageButtons() {
    document.getElementById('switch-to-en').classList.toggle('active', currentLanguage === 'en');
    document.getElementById('switch-to-zh').classList.toggle('active', currentLanguage === 'zh');
}

// 应用语言到页面
function applyLanguage() {
    // 获取当前语言的翻译
    const lang = translations[currentLanguage];
    
    // 更新所有带有data-i18n属性的元素
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (lang[key]) {
            if (element.tagName === 'INPUT' && element.type === 'text') {
                element.placeholder = lang[key];
            } else {
                element.textContent = lang[key];
            }
        }
    });
    
    // 更新所有带有data-i18n-html属性的元素（用于HTML内容）
    document.querySelectorAll('[data-i18n-html]').forEach(element => {
        const key = element.getAttribute('data-i18n-html');
        if (lang[key]) {
            element.innerHTML = lang[key];
        }
    });
    
    // 保存语言偏好
    localStorage.setItem('preferredLanguage', currentLanguage);
}

// 切换语言
function switchLanguage(lang) {
    currentLanguage = lang;
    updateLanguageButtons();
    applyLanguage();
}

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initLanguage();
    
    // 添加事件监听器
    document.getElementById('switch-to-en').addEventListener('click', function() {
        switchLanguage('en');
    });
    
    document.getElementById('switch-to-zh').addEventListener('click', function() {
        switchLanguage('zh');
    });
});