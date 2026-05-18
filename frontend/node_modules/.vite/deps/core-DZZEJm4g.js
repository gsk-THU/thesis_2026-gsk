import { t as __commonJSMin } from "./chunk-BoAXSpZd.js";
//#region node_modules/highlight.js/lib/core.js
var require_core$1 = /* @__PURE__ */ __commonJSMin(((exports, module) => {
	function deepFreeze(obj) {
		if (obj instanceof Map) obj.clear = obj.delete = obj.set = function() {
			throw new Error("map is read-only");
		};
		else if (obj instanceof Set) obj.add = obj.clear = obj.delete = function() {
			throw new Error("set is read-only");
		};
		Object.freeze(obj);
		Object.getOwnPropertyNames(obj).forEach(function(name) {
			var prop = obj[name];
			if (typeof prop == "object" && !Object.isFrozen(prop)) deepFreeze(prop);
		});
		return obj;
	}
	var deepFreezeEs6 = deepFreeze;
	deepFreezeEs6.default = deepFreeze;
	/** @implements CallbackResponse */
	var Response = class {
		/**
		* @param {CompiledMode} mode
		*/
		constructor(mode) {
			if (mode.data === void 0) mode.data = {};
			this.data = mode.data;
			this.isMatchIgnored = false;
		}
		ignoreMatch() {
			this.isMatchIgnored = true;
		}
	};
	/**
	* @param {string} value
	* @returns {string}
	*/
	function escapeHTML(value) {
		return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
	}
	/**
	* performs a shallow merge of multiple objects into one
	*
	* @template T
	* @param {T} original
	* @param {Record<string,any>[]} objects
	* @returns {T} a single new object
	*/
	function inherit(original, ...objects) {
		/** @type Record<string,any> */
		const result = Object.create(null);
		for (const key in original) result[key] = original[key];
		objects.forEach(function(obj) {
			for (const key in obj) result[key] = obj[key];
		});
		return result;
	}
	/**
	* @typedef {object} Renderer
	* @property {(text: string) => void} addText
	* @property {(node: Node) => void} openNode
	* @property {(node: Node) => void} closeNode
	* @property {() => string} value
	*/
	/** @typedef {{kind?: string, sublanguage?: boolean}} Node */
	/** @typedef {{walk: (r: Renderer) => void}} Tree */
	/** */
	var SPAN_CLOSE = "</span>";
	/**
	* Determines if a node needs to be wrapped in <span>
	*
	* @param {Node} node */
	var emitsWrappingTags = (node) => {
		return !!node.kind;
	};
	/** @type {Renderer} */
	var HTMLRenderer = class {
		/**
		* Creates a new HTMLRenderer
		*
		* @param {Tree} parseTree - the parse tree (must support `walk` API)
		* @param {{classPrefix: string}} options
		*/
		constructor(parseTree, options) {
			this.buffer = "";
			this.classPrefix = options.classPrefix;
			parseTree.walk(this);
		}
		/**
		* Adds texts to the output stream
		*
		* @param {string} text */
		addText(text) {
			this.buffer += escapeHTML(text);
		}
		/**
		* Adds a node open to the output stream (if needed)
		*
		* @param {Node} node */
		openNode(node) {
			if (!emitsWrappingTags(node)) return;
			let className = node.kind;
			if (!node.sublanguage) className = `${this.classPrefix}${className}`;
			this.span(className);
		}
		/**
		* Adds a node close to the output stream (if needed)
		*
		* @param {Node} node */
		closeNode(node) {
			if (!emitsWrappingTags(node)) return;
			this.buffer += SPAN_CLOSE;
		}
		/**
		* returns the accumulated buffer
		*/
		value() {
			return this.buffer;
		}
		/**
		* Builds a span element
		*
		* @param {string} className */
		span(className) {
			this.buffer += `<span class="${className}">`;
		}
	};
	/** @typedef {{kind?: string, sublanguage?: boolean, children: Node[]} | string} Node */
	/** @typedef {{kind?: string, sublanguage?: boolean, children: Node[]} } DataNode */
	/**  */
	var TokenTree = class TokenTree {
		constructor() {
			/** @type DataNode */
			this.rootNode = { children: [] };
			this.stack = [this.rootNode];
		}
		get top() {
			return this.stack[this.stack.length - 1];
		}
		get root() {
			return this.rootNode;
		}
		/** @param {Node} node */
		add(node) {
			this.top.children.push(node);
		}
		/** @param {string} kind */
		openNode(kind) {
			/** @type Node */
			const node = {
				kind,
				children: []
			};
			this.add(node);
			this.stack.push(node);
		}
		closeNode() {
			if (this.stack.length > 1) return this.stack.pop();
		}
		closeAllNodes() {
			while (this.closeNode());
		}
		toJSON() {
			return JSON.stringify(this.rootNode, null, 4);
		}
		/**
		* @typedef { import("./html_renderer").Renderer } Renderer
		* @param {Renderer} builder
		*/
		walk(builder) {
			return this.constructor._walk(builder, this.rootNode);
		}
		/**
		* @param {Renderer} builder
		* @param {Node} node
		*/
		static _walk(builder, node) {
			if (typeof node === "string") builder.addText(node);
			else if (node.children) {
				builder.openNode(node);
				node.children.forEach((child) => this._walk(builder, child));
				builder.closeNode(node);
			}
			return builder;
		}
		/**
		* @param {Node} node
		*/
		static _collapse(node) {
			if (typeof node === "string") return;
			if (!node.children) return;
			if (node.children.every((el) => typeof el === "string")) node.children = [node.children.join("")];
			else node.children.forEach((child) => {
				TokenTree._collapse(child);
			});
		}
	};
	/**
	Currently this is all private API, but this is the minimal API necessary
	that an Emitter must implement to fully support the parser.
	
	Minimal interface:
	
	- addKeyword(text, kind)
	- addText(text)
	- addSublanguage(emitter, subLanguageName)
	- finalize()
	- openNode(kind)
	- closeNode()
	- closeAllNodes()
	- toHTML()
	
	*/
	/**
	* @implements {Emitter}
	*/
	var TokenTreeEmitter = class extends TokenTree {
		/**
		* @param {*} options
		*/
		constructor(options) {
			super();
			this.options = options;
		}
		/**
		* @param {string} text
		* @param {string} kind
		*/
		addKeyword(text, kind) {
			if (text === "") return;
			this.openNode(kind);
			this.addText(text);
			this.closeNode();
		}
		/**
		* @param {string} text
		*/
		addText(text) {
			if (text === "") return;
			this.add(text);
		}
		/**
		* @param {Emitter & {root: DataNode}} emitter
		* @param {string} name
		*/
		addSublanguage(emitter, name) {
			/** @type DataNode */
			const node = emitter.root;
			node.kind = name;
			node.sublanguage = true;
			this.add(node);
		}
		toHTML() {
			return new HTMLRenderer(this, this.options).value();
		}
		finalize() {
			return true;
		}
	};
	/**
	* @param {string} value
	* @returns {RegExp}
	* */
	function escape(value) {
		return new RegExp(value.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&"), "m");
	}
	/**
	* @param {RegExp | string } re
	* @returns {string}
	*/
	function source(re) {
		if (!re) return null;
		if (typeof re === "string") return re;
		return re.source;
	}
	/**
	* @param {...(RegExp | string) } args
	* @returns {string}
	*/
	function concat(...args) {
		return args.map((x) => source(x)).join("");
	}
	/**
	* Any of the passed expresssions may match
	*
	* Creates a huge this | this | that | that match
	* @param {(RegExp | string)[] } args
	* @returns {string}
	*/
	function either(...args) {
		return "(" + args.map((x) => source(x)).join("|") + ")";
	}
	/**
	* @param {RegExp} re
	* @returns {number}
	*/
	function countMatchGroups(re) {
		return new RegExp(re.toString() + "|").exec("").length - 1;
	}
	/**
	* Does lexeme start with a regular expression match at the beginning
	* @param {RegExp} re
	* @param {string} lexeme
	*/
	function startsWith(re, lexeme) {
		const match = re && re.exec(lexeme);
		return match && match.index === 0;
	}
	var BACKREF_RE = /\[(?:[^\\\]]|\\.)*\]|\(\??|\\([1-9][0-9]*)|\\./;
	/**
	* @param {(string | RegExp)[]} regexps
	* @param {string} separator
	* @returns {string}
	*/
	function join(regexps, separator = "|") {
		let numCaptures = 0;
		return regexps.map((regex) => {
			numCaptures += 1;
			const offset = numCaptures;
			let re = source(regex);
			let out = "";
			while (re.length > 0) {
				const match = BACKREF_RE.exec(re);
				if (!match) {
					out += re;
					break;
				}
				out += re.substring(0, match.index);
				re = re.substring(match.index + match[0].length);
				if (match[0][0] === "\\" && match[1]) out += "\\" + String(Number(match[1]) + offset);
				else {
					out += match[0];
					if (match[0] === "(") numCaptures++;
				}
			}
			return out;
		}).map((re) => `(${re})`).join(separator);
	}
	var MATCH_NOTHING_RE = /\b\B/;
	var IDENT_RE = "[a-zA-Z]\\w*";
	var UNDERSCORE_IDENT_RE = "[a-zA-Z_]\\w*";
	var NUMBER_RE = "\\b\\d+(\\.\\d+)?";
	var C_NUMBER_RE = "(-?)(\\b0[xX][a-fA-F0-9]+|(\\b\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?)";
	var BINARY_NUMBER_RE = "\\b(0b[01]+)";
	var RE_STARTERS_RE = "!|!=|!==|%|%=|&|&&|&=|\\*|\\*=|\\+|\\+=|,|-|-=|/=|/|:|;|<<|<<=|<=|<|===|==|=|>>>=|>>=|>=|>>>|>>|>|\\?|\\[|\\{|\\(|\\^|\\^=|\\||\\|=|\\|\\||~";
	/**
	* @param { Partial<Mode> & {binary?: string | RegExp} } opts
	*/
	var SHEBANG = (opts = {}) => {
		const beginShebang = /^#![ ]*\//;
		if (opts.binary) opts.begin = concat(beginShebang, /.*\b/, opts.binary, /\b.*/);
		return inherit({
			className: "meta",
			begin: beginShebang,
			end: /$/,
			relevance: 0,
			"on:begin": (m, resp) => {
				if (m.index !== 0) resp.ignoreMatch();
			}
		}, opts);
	};
	var BACKSLASH_ESCAPE = {
		begin: "\\\\[\\s\\S]",
		relevance: 0
	};
	var APOS_STRING_MODE = {
		className: "string",
		begin: "'",
		end: "'",
		illegal: "\\n",
		contains: [BACKSLASH_ESCAPE]
	};
	var QUOTE_STRING_MODE = {
		className: "string",
		begin: "\"",
		end: "\"",
		illegal: "\\n",
		contains: [BACKSLASH_ESCAPE]
	};
	var PHRASAL_WORDS_MODE = { begin: /\b(a|an|the|are|I'm|isn't|don't|doesn't|won't|but|just|should|pretty|simply|enough|gonna|going|wtf|so|such|will|you|your|they|like|more)\b/ };
	/**
	* Creates a comment mode
	*
	* @param {string | RegExp} begin
	* @param {string | RegExp} end
	* @param {Mode | {}} [modeOptions]
	* @returns {Partial<Mode>}
	*/
	var COMMENT = function(begin, end, modeOptions = {}) {
		const mode = inherit({
			className: "comment",
			begin,
			end,
			contains: []
		}, modeOptions);
		mode.contains.push(PHRASAL_WORDS_MODE);
		mode.contains.push({
			className: "doctag",
			begin: "(?:TODO|FIXME|NOTE|BUG|OPTIMIZE|HACK|XXX):",
			relevance: 0
		});
		return mode;
	};
	var C_LINE_COMMENT_MODE = COMMENT("//", "$");
	var C_BLOCK_COMMENT_MODE = COMMENT("/\\*", "\\*/");
	var HASH_COMMENT_MODE = COMMENT("#", "$");
	var NUMBER_MODE = {
		className: "number",
		begin: NUMBER_RE,
		relevance: 0
	};
	var C_NUMBER_MODE = {
		className: "number",
		begin: C_NUMBER_RE,
		relevance: 0
	};
	var BINARY_NUMBER_MODE = {
		className: "number",
		begin: BINARY_NUMBER_RE,
		relevance: 0
	};
	var CSS_NUMBER_MODE = {
		className: "number",
		begin: NUMBER_RE + "(%|em|ex|ch|rem|vw|vh|vmin|vmax|cm|mm|in|pt|pc|px|deg|grad|rad|turn|s|ms|Hz|kHz|dpi|dpcm|dppx)?",
		relevance: 0
	};
	var REGEXP_MODE = {
		begin: /(?=\/[^/\n]*\/)/,
		contains: [{
			className: "regexp",
			begin: /\//,
			end: /\/[gimuy]*/,
			illegal: /\n/,
			contains: [BACKSLASH_ESCAPE, {
				begin: /\[/,
				end: /\]/,
				relevance: 0,
				contains: [BACKSLASH_ESCAPE]
			}]
		}]
	};
	var TITLE_MODE = {
		className: "title",
		begin: IDENT_RE,
		relevance: 0
	};
	var UNDERSCORE_TITLE_MODE = {
		className: "title",
		begin: UNDERSCORE_IDENT_RE,
		relevance: 0
	};
	var METHOD_GUARD = {
		begin: "\\.\\s*" + UNDERSCORE_IDENT_RE,
		relevance: 0
	};
	/**
	* Adds end same as begin mechanics to a mode
	*
	* Your mode must include at least a single () match group as that first match
	* group is what is used for comparison
	* @param {Partial<Mode>} mode
	*/
	var END_SAME_AS_BEGIN = function(mode) {
		return Object.assign(mode, {
			"on:begin": (m, resp) => {
				resp.data._beginMatch = m[1];
			},
			"on:end": (m, resp) => {
				if (resp.data._beginMatch !== m[1]) resp.ignoreMatch();
			}
		});
	};
	var MODES = /* @__PURE__ */ Object.freeze({
		__proto__: null,
		MATCH_NOTHING_RE,
		IDENT_RE,
		UNDERSCORE_IDENT_RE,
		NUMBER_RE,
		C_NUMBER_RE,
		BINARY_NUMBER_RE,
		RE_STARTERS_RE,
		SHEBANG,
		BACKSLASH_ESCAPE,
		APOS_STRING_MODE,
		QUOTE_STRING_MODE,
		PHRASAL_WORDS_MODE,
		COMMENT,
		C_LINE_COMMENT_MODE,
		C_BLOCK_COMMENT_MODE,
		HASH_COMMENT_MODE,
		NUMBER_MODE,
		C_NUMBER_MODE,
		BINARY_NUMBER_MODE,
		CSS_NUMBER_MODE,
		REGEXP_MODE,
		TITLE_MODE,
		UNDERSCORE_TITLE_MODE,
		METHOD_GUARD,
		END_SAME_AS_BEGIN
	});
	/**
	* Skip a match if it has a preceding dot
	*
	* This is used for `beginKeywords` to prevent matching expressions such as
	* `bob.keyword.do()`. The mode compiler automatically wires this up as a
	* special _internal_ 'on:begin' callback for modes with `beginKeywords`
	* @param {RegExpMatchArray} match
	* @param {CallbackResponse} response
	*/
	function skipIfhasPrecedingDot(match, response) {
		if (match.input[match.index - 1] === ".") response.ignoreMatch();
	}
	/**
	* `beginKeywords` syntactic sugar
	* @type {CompilerExt}
	*/
	function beginKeywords(mode, parent) {
		if (!parent) return;
		if (!mode.beginKeywords) return;
		mode.begin = "\\b(" + mode.beginKeywords.split(" ").join("|") + ")(?!\\.)(?=\\b|\\s)";
		mode.__beforeBegin = skipIfhasPrecedingDot;
		mode.keywords = mode.keywords || mode.beginKeywords;
		delete mode.beginKeywords;
		if (mode.relevance === void 0) mode.relevance = 0;
	}
	/**
	* Allow `illegal` to contain an array of illegal values
	* @type {CompilerExt}
	*/
	function compileIllegal(mode, _parent) {
		if (!Array.isArray(mode.illegal)) return;
		mode.illegal = either(...mode.illegal);
	}
	/**
	* `match` to match a single expression for readability
	* @type {CompilerExt}
	*/
	function compileMatch(mode, _parent) {
		if (!mode.match) return;
		if (mode.begin || mode.end) throw new Error("begin & end are not supported with match");
		mode.begin = mode.match;
		delete mode.match;
	}
	/**
	* provides the default 1 relevance to all modes
	* @type {CompilerExt}
	*/
	function compileRelevance(mode, _parent) {
		if (mode.relevance === void 0) mode.relevance = 1;
	}
	var COMMON_KEYWORDS = [
		"of",
		"and",
		"for",
		"in",
		"not",
		"or",
		"if",
		"then",
		"parent",
		"list",
		"value"
	];
	var DEFAULT_KEYWORD_CLASSNAME = "keyword";
	/**
	* Given raw keywords from a language definition, compile them.
	*
	* @param {string | Record<string,string|string[]> | Array<string>} rawKeywords
	* @param {boolean} caseInsensitive
	*/
	function compileKeywords(rawKeywords, caseInsensitive, className = DEFAULT_KEYWORD_CLASSNAME) {
		/** @type KeywordDict */
		const compiledKeywords = {};
		if (typeof rawKeywords === "string") compileList(className, rawKeywords.split(" "));
		else if (Array.isArray(rawKeywords)) compileList(className, rawKeywords);
		else Object.keys(rawKeywords).forEach(function(className) {
			Object.assign(compiledKeywords, compileKeywords(rawKeywords[className], caseInsensitive, className));
		});
		return compiledKeywords;
		/**
		* Compiles an individual list of keywords
		*
		* Ex: "for if when while|5"
		*
		* @param {string} className
		* @param {Array<string>} keywordList
		*/
		function compileList(className, keywordList) {
			if (caseInsensitive) keywordList = keywordList.map((x) => x.toLowerCase());
			keywordList.forEach(function(keyword) {
				const pair = keyword.split("|");
				compiledKeywords[pair[0]] = [className, scoreForKeyword(pair[0], pair[1])];
			});
		}
	}
	/**
	* Returns the proper score for a given keyword
	*
	* Also takes into account comment keywords, which will be scored 0 UNLESS
	* another score has been manually assigned.
	* @param {string} keyword
	* @param {string} [providedScore]
	*/
	function scoreForKeyword(keyword, providedScore) {
		if (providedScore) return Number(providedScore);
		return commonKeyword(keyword) ? 0 : 1;
	}
	/**
	* Determines if a given keyword is common or not
	*
	* @param {string} keyword */
	function commonKeyword(keyword) {
		return COMMON_KEYWORDS.includes(keyword.toLowerCase());
	}
	/**
	* Compiles a language definition result
	*
	* Given the raw result of a language definition (Language), compiles this so
	* that it is ready for highlighting code.
	* @param {Language} language
	* @param {{plugins: HLJSPlugin[]}} opts
	* @returns {CompiledLanguage}
	*/
	function compileLanguage(language, { plugins }) {
		/**
		* Builds a regex with the case sensativility of the current language
		*
		* @param {RegExp | string} value
		* @param {boolean} [global]
		*/
		function langRe(value, global) {
			return new RegExp(source(value), "m" + (language.case_insensitive ? "i" : "") + (global ? "g" : ""));
		}
		/**
		Stores multiple regular expressions and allows you to quickly search for
		them all in a string simultaneously - returning the first match.  It does
		this by creating a huge (a|b|c) regex - each individual item wrapped with ()
		and joined by `|` - using match groups to track position.  When a match is
		found checking which position in the array has content allows us to figure
		out which of the original regexes / match groups triggered the match.
		
		The match object itself (the result of `Regex.exec`) is returned but also
		enhanced by merging in any meta-data that was registered with the regex.
		This is how we keep track of which mode matched, and what type of rule
		(`illegal`, `begin`, end, etc).
		*/
		class MultiRegex {
			constructor() {
				this.matchIndexes = {};
				this.regexes = [];
				this.matchAt = 1;
				this.position = 0;
			}
			addRule(re, opts) {
				opts.position = this.position++;
				this.matchIndexes[this.matchAt] = opts;
				this.regexes.push([opts, re]);
				this.matchAt += countMatchGroups(re) + 1;
			}
			compile() {
				if (this.regexes.length === 0) this.exec = () => null;
				this.matcherRe = langRe(join(this.regexes.map((el) => el[1])), true);
				this.lastIndex = 0;
			}
			/** @param {string} s */
			exec(s) {
				this.matcherRe.lastIndex = this.lastIndex;
				const match = this.matcherRe.exec(s);
				if (!match) return null;
				const i = match.findIndex((el, i) => i > 0 && el !== void 0);
				const matchData = this.matchIndexes[i];
				match.splice(0, i);
				return Object.assign(match, matchData);
			}
		}
		class ResumableMultiRegex {
			constructor() {
				this.rules = [];
				this.multiRegexes = [];
				this.count = 0;
				this.lastIndex = 0;
				this.regexIndex = 0;
			}
			getMatcher(index) {
				if (this.multiRegexes[index]) return this.multiRegexes[index];
				const matcher = new MultiRegex();
				this.rules.slice(index).forEach(([re, opts]) => matcher.addRule(re, opts));
				matcher.compile();
				this.multiRegexes[index] = matcher;
				return matcher;
			}
			resumingScanAtSamePosition() {
				return this.regexIndex !== 0;
			}
			considerAll() {
				this.regexIndex = 0;
			}
			addRule(re, opts) {
				this.rules.push([re, opts]);
				if (opts.type === "begin") this.count++;
			}
			/** @param {string} s */
			exec(s) {
				const m = this.getMatcher(this.regexIndex);
				m.lastIndex = this.lastIndex;
				let result = m.exec(s);
				if (this.resumingScanAtSamePosition()) if (result && result.index === this.lastIndex);
				else {
					const m2 = this.getMatcher(0);
					m2.lastIndex = this.lastIndex + 1;
					result = m2.exec(s);
				}
				if (result) {
					this.regexIndex += result.position + 1;
					if (this.regexIndex === this.count) this.considerAll();
				}
				return result;
			}
		}
		/**
		* Given a mode, builds a huge ResumableMultiRegex that can be used to walk
		* the content and find matches.
		*
		* @param {CompiledMode} mode
		* @returns {ResumableMultiRegex}
		*/
		function buildModeRegex(mode) {
			const mm = new ResumableMultiRegex();
			mode.contains.forEach((term) => mm.addRule(term.begin, {
				rule: term,
				type: "begin"
			}));
			if (mode.terminatorEnd) mm.addRule(mode.terminatorEnd, { type: "end" });
			if (mode.illegal) mm.addRule(mode.illegal, { type: "illegal" });
			return mm;
		}
		/** skip vs abort vs ignore
		*
		* @skip   - The mode is still entered and exited normally (and contains rules apply),
		*           but all content is held and added to the parent buffer rather than being
		*           output when the mode ends.  Mostly used with `sublanguage` to build up
		*           a single large buffer than can be parsed by sublanguage.
		*
		*             - The mode begin ands ends normally.
		*             - Content matched is added to the parent mode buffer.
		*             - The parser cursor is moved forward normally.
		*
		* @abort  - A hack placeholder until we have ignore.  Aborts the mode (as if it
		*           never matched) but DOES NOT continue to match subsequent `contains`
		*           modes.  Abort is bad/suboptimal because it can result in modes
		*           farther down not getting applied because an earlier rule eats the
		*           content but then aborts.
		*
		*             - The mode does not begin.
		*             - Content matched by `begin` is added to the mode buffer.
		*             - The parser cursor is moved forward accordingly.
		*
		* @ignore - Ignores the mode (as if it never matched) and continues to match any
		*           subsequent `contains` modes.  Ignore isn't technically possible with
		*           the current parser implementation.
		*
		*             - The mode does not begin.
		*             - Content matched by `begin` is ignored.
		*             - The parser cursor is not moved forward.
		*/
		/**
		* Compiles an individual mode
		*
		* This can raise an error if the mode contains certain detectable known logic
		* issues.
		* @param {Mode} mode
		* @param {CompiledMode | null} [parent]
		* @returns {CompiledMode | never}
		*/
		function compileMode(mode, parent) {
			const cmode = mode;
			if (mode.isCompiled) return cmode;
			[compileMatch].forEach((ext) => ext(mode, parent));
			language.compilerExtensions.forEach((ext) => ext(mode, parent));
			mode.__beforeBegin = null;
			[
				beginKeywords,
				compileIllegal,
				compileRelevance
			].forEach((ext) => ext(mode, parent));
			mode.isCompiled = true;
			let keywordPattern = null;
			if (typeof mode.keywords === "object") {
				keywordPattern = mode.keywords.$pattern;
				delete mode.keywords.$pattern;
			}
			if (mode.keywords) mode.keywords = compileKeywords(mode.keywords, language.case_insensitive);
			if (mode.lexemes && keywordPattern) throw new Error("ERR: Prefer `keywords.$pattern` to `mode.lexemes`, BOTH are not allowed. (see mode reference) ");
			keywordPattern = keywordPattern || mode.lexemes || /\w+/;
			cmode.keywordPatternRe = langRe(keywordPattern, true);
			if (parent) {
				if (!mode.begin) mode.begin = /\B|\b/;
				cmode.beginRe = langRe(mode.begin);
				if (mode.endSameAsBegin) mode.end = mode.begin;
				if (!mode.end && !mode.endsWithParent) mode.end = /\B|\b/;
				if (mode.end) cmode.endRe = langRe(mode.end);
				cmode.terminatorEnd = source(mode.end) || "";
				if (mode.endsWithParent && parent.terminatorEnd) cmode.terminatorEnd += (mode.end ? "|" : "") + parent.terminatorEnd;
			}
			if (mode.illegal) cmode.illegalRe = langRe(mode.illegal);
			if (!mode.contains) mode.contains = [];
			mode.contains = [].concat(...mode.contains.map(function(c) {
				return expandOrCloneMode(c === "self" ? mode : c);
			}));
			mode.contains.forEach(function(c) {
				compileMode(c, cmode);
			});
			if (mode.starts) compileMode(mode.starts, parent);
			cmode.matcher = buildModeRegex(cmode);
			return cmode;
		}
		if (!language.compilerExtensions) language.compilerExtensions = [];
		if (language.contains && language.contains.includes("self")) throw new Error("ERR: contains `self` is not supported at the top-level of a language.  See documentation.");
		language.classNameAliases = inherit(language.classNameAliases || {});
		return compileMode(language);
	}
	/**
	* Determines if a mode has a dependency on it's parent or not
	*
	* If a mode does have a parent dependency then often we need to clone it if
	* it's used in multiple places so that each copy points to the correct parent,
	* where-as modes without a parent can often safely be re-used at the bottom of
	* a mode chain.
	*
	* @param {Mode | null} mode
	* @returns {boolean} - is there a dependency on the parent?
	* */
	function dependencyOnParent(mode) {
		if (!mode) return false;
		return mode.endsWithParent || dependencyOnParent(mode.starts);
	}
	/**
	* Expands a mode or clones it if necessary
	*
	* This is necessary for modes with parental dependenceis (see notes on
	* `dependencyOnParent`) and for nodes that have `variants` - which must then be
	* exploded into their own individual modes at compile time.
	*
	* @param {Mode} mode
	* @returns {Mode | Mode[]}
	* */
	function expandOrCloneMode(mode) {
		if (mode.variants && !mode.cachedVariants) mode.cachedVariants = mode.variants.map(function(variant) {
			return inherit(mode, { variants: null }, variant);
		});
		if (mode.cachedVariants) return mode.cachedVariants;
		if (dependencyOnParent(mode)) return inherit(mode, { starts: mode.starts ? inherit(mode.starts) : null });
		if (Object.isFrozen(mode)) return inherit(mode);
		return mode;
	}
	var version = "10.7.3";
	function hasValueOrEmptyAttribute(value) {
		return Boolean(value || value === "");
	}
	function BuildVuePlugin(hljs) {
		const Component = {
			props: [
				"language",
				"code",
				"autodetect"
			],
			data: function() {
				return {
					detectedLanguage: "",
					unknownLanguage: false
				};
			},
			computed: {
				className() {
					if (this.unknownLanguage) return "";
					return "hljs " + this.detectedLanguage;
				},
				highlighted() {
					if (!this.autoDetect && !hljs.getLanguage(this.language)) {
						console.warn(`The language "${this.language}" you specified could not be found.`);
						this.unknownLanguage = true;
						return escapeHTML(this.code);
					}
					let result = {};
					if (this.autoDetect) {
						result = hljs.highlightAuto(this.code);
						this.detectedLanguage = result.language;
					} else {
						result = hljs.highlight(this.language, this.code, this.ignoreIllegals);
						this.detectedLanguage = this.language;
					}
					return result.value;
				},
				autoDetect() {
					return !this.language || hasValueOrEmptyAttribute(this.autodetect);
				},
				ignoreIllegals() {
					return true;
				}
			},
			render(createElement) {
				return createElement("pre", {}, [createElement("code", {
					class: this.className,
					domProps: { innerHTML: this.highlighted }
				})]);
			}
		};
		return {
			Component,
			VuePlugin: { install(Vue) {
				Vue.component("highlightjs", Component);
			} }
		};
	}
	/** @type {HLJSPlugin} */
	var mergeHTMLPlugin = { "after:highlightElement": ({ el, result, text }) => {
		const originalStream = nodeStream(el);
		if (!originalStream.length) return;
		const resultNode = document.createElement("div");
		resultNode.innerHTML = result.value;
		result.value = mergeStreams(originalStream, nodeStream(resultNode), text);
	} };
	/**
	* @typedef Event
	* @property {'start'|'stop'} event
	* @property {number} offset
	* @property {Node} node
	*/
	/**
	* @param {Node} node
	*/
	function tag(node) {
		return node.nodeName.toLowerCase();
	}
	/**
	* @param {Node} node
	*/
	function nodeStream(node) {
		/** @type Event[] */
		const result = [];
		(function _nodeStream(node, offset) {
			for (let child = node.firstChild; child; child = child.nextSibling) if (child.nodeType === 3) offset += child.nodeValue.length;
			else if (child.nodeType === 1) {
				result.push({
					event: "start",
					offset,
					node: child
				});
				offset = _nodeStream(child, offset);
				if (!tag(child).match(/br|hr|img|input/)) result.push({
					event: "stop",
					offset,
					node: child
				});
			}
			return offset;
		})(node, 0);
		return result;
	}
	/**
	* @param {any} original - the original stream
	* @param {any} highlighted - stream of the highlighted source
	* @param {string} value - the original source itself
	*/
	function mergeStreams(original, highlighted, value) {
		let processed = 0;
		let result = "";
		const nodeStack = [];
		function selectStream() {
			if (!original.length || !highlighted.length) return original.length ? original : highlighted;
			if (original[0].offset !== highlighted[0].offset) return original[0].offset < highlighted[0].offset ? original : highlighted;
			return highlighted[0].event === "start" ? original : highlighted;
		}
		/**
		* @param {Node} node
		*/
		function open(node) {
			/** @param {Attr} attr */
			function attributeString(attr) {
				return " " + attr.nodeName + "=\"" + escapeHTML(attr.value) + "\"";
			}
			result += "<" + tag(node) + [].map.call(node.attributes, attributeString).join("") + ">";
		}
		/**
		* @param {Node} node
		*/
		function close(node) {
			result += "</" + tag(node) + ">";
		}
		/**
		* @param {Event} event
		*/
		function render(event) {
			(event.event === "start" ? open : close)(event.node);
		}
		while (original.length || highlighted.length) {
			let stream = selectStream();
			result += escapeHTML(value.substring(processed, stream[0].offset));
			processed = stream[0].offset;
			if (stream === original) {
				nodeStack.reverse().forEach(close);
				do {
					render(stream.splice(0, 1)[0]);
					stream = selectStream();
				} while (stream === original && stream.length && stream[0].offset === processed);
				nodeStack.reverse().forEach(open);
			} else {
				if (stream[0].event === "start") nodeStack.push(stream[0].node);
				else nodeStack.pop();
				render(stream.splice(0, 1)[0]);
			}
		}
		return result + escapeHTML(value.substr(processed));
	}
	/**
	* @type {Record<string, boolean>}
	*/
	var seenDeprecations = {};
	/**
	* @param {string} message
	*/
	var error = (message) => {
		console.error(message);
	};
	/**
	* @param {string} message
	* @param {any} args
	*/
	var warn = (message, ...args) => {
		console.log(`WARN: ${message}`, ...args);
	};
	/**
	* @param {string} version
	* @param {string} message
	*/
	var deprecated = (version, message) => {
		if (seenDeprecations[`${version}/${message}`]) return;
		console.log(`Deprecated as of ${version}. ${message}`);
		seenDeprecations[`${version}/${message}`] = true;
	};
	var escape$1 = escapeHTML;
	var inherit$1 = inherit;
	var NO_MATCH = Symbol("nomatch");
	/**
	* @param {any} hljs - object that is extended (legacy)
	* @returns {HLJSApi}
	*/
	var HLJS = function(hljs) {
		/** @type {Record<string, Language>} */
		const languages = Object.create(null);
		/** @type {Record<string, string>} */
		const aliases = Object.create(null);
		/** @type {HLJSPlugin[]} */
		const plugins = [];
		let SAFE_MODE = true;
		const fixMarkupRe = /(^(<[^>]+>|\t|)+|\n)/gm;
		const LANGUAGE_NOT_FOUND = "Could not find the language '{}', did you forget to load/include a language module?";
		/** @type {Language} */
		const PLAINTEXT_LANGUAGE = {
			disableAutodetect: true,
			name: "Plain text",
			contains: []
		};
		/** @type HLJSOptions */
		let options = {
			noHighlightRe: /^(no-?highlight)$/i,
			languageDetectRe: /\blang(?:uage)?-([\w-]+)\b/i,
			classPrefix: "hljs-",
			tabReplace: null,
			useBR: false,
			languages: null,
			__emitter: TokenTreeEmitter
		};
		/**
		* Tests a language name to see if highlighting should be skipped
		* @param {string} languageName
		*/
		function shouldNotHighlight(languageName) {
			return options.noHighlightRe.test(languageName);
		}
		/**
		* @param {HighlightedHTMLElement} block - the HTML element to determine language for
		*/
		function blockLanguage(block) {
			let classes = block.className + " ";
			classes += block.parentNode ? block.parentNode.className : "";
			const match = options.languageDetectRe.exec(classes);
			if (match) {
				const language = getLanguage(match[1]);
				if (!language) {
					warn(LANGUAGE_NOT_FOUND.replace("{}", match[1]));
					warn("Falling back to no-highlight mode for this block.", block);
				}
				return language ? match[1] : "no-highlight";
			}
			return classes.split(/\s+/).find((_class) => shouldNotHighlight(_class) || getLanguage(_class));
		}
		/**
		* Core highlighting function.
		*
		* OLD API
		* highlight(lang, code, ignoreIllegals, continuation)
		*
		* NEW API
		* highlight(code, {lang, ignoreIllegals})
		*
		* @param {string} codeOrlanguageName - the language to use for highlighting
		* @param {string | HighlightOptions} optionsOrCode - the code to highlight
		* @param {boolean} [ignoreIllegals] - whether to ignore illegal matches, default is to bail
		* @param {CompiledMode} [continuation] - current continuation mode, if any
		*
		* @returns {HighlightResult} Result - an object that represents the result
		* @property {string} language - the language name
		* @property {number} relevance - the relevance score
		* @property {string} value - the highlighted HTML code
		* @property {string} code - the original raw code
		* @property {CompiledMode} top - top of the current mode stack
		* @property {boolean} illegal - indicates whether any illegal matches were found
		*/
		function highlight(codeOrlanguageName, optionsOrCode, ignoreIllegals, continuation) {
			let code = "";
			let languageName = "";
			if (typeof optionsOrCode === "object") {
				code = codeOrlanguageName;
				ignoreIllegals = optionsOrCode.ignoreIllegals;
				languageName = optionsOrCode.language;
				continuation = void 0;
			} else {
				deprecated("10.7.0", "highlight(lang, code, ...args) has been deprecated.");
				deprecated("10.7.0", "Please use highlight(code, options) instead.\nhttps://github.com/highlightjs/highlight.js/issues/2277");
				languageName = codeOrlanguageName;
				code = optionsOrCode;
			}
			/** @type {BeforeHighlightContext} */
			const context = {
				code,
				language: languageName
			};
			fire("before:highlight", context);
			const result = context.result ? context.result : _highlight(context.language, context.code, ignoreIllegals, continuation);
			result.code = context.code;
			fire("after:highlight", result);
			return result;
		}
		/**
		* private highlight that's used internally and does not fire callbacks
		*
		* @param {string} languageName - the language to use for highlighting
		* @param {string} codeToHighlight - the code to highlight
		* @param {boolean?} [ignoreIllegals] - whether to ignore illegal matches, default is to bail
		* @param {CompiledMode?} [continuation] - current continuation mode, if any
		* @returns {HighlightResult} - result of the highlight operation
		*/
		function _highlight(languageName, codeToHighlight, ignoreIllegals, continuation) {
			/**
			* Return keyword data if a match is a keyword
			* @param {CompiledMode} mode - current mode
			* @param {RegExpMatchArray} match - regexp match data
			* @returns {KeywordData | false}
			*/
			function keywordData(mode, match) {
				const matchText = language.case_insensitive ? match[0].toLowerCase() : match[0];
				return Object.prototype.hasOwnProperty.call(mode.keywords, matchText) && mode.keywords[matchText];
			}
			function processKeywords() {
				if (!top.keywords) {
					emitter.addText(modeBuffer);
					return;
				}
				let lastIndex = 0;
				top.keywordPatternRe.lastIndex = 0;
				let match = top.keywordPatternRe.exec(modeBuffer);
				let buf = "";
				while (match) {
					buf += modeBuffer.substring(lastIndex, match.index);
					const data = keywordData(top, match);
					if (data) {
						const [kind, keywordRelevance] = data;
						emitter.addText(buf);
						buf = "";
						relevance += keywordRelevance;
						if (kind.startsWith("_")) buf += match[0];
						else {
							const cssClass = language.classNameAliases[kind] || kind;
							emitter.addKeyword(match[0], cssClass);
						}
					} else buf += match[0];
					lastIndex = top.keywordPatternRe.lastIndex;
					match = top.keywordPatternRe.exec(modeBuffer);
				}
				buf += modeBuffer.substr(lastIndex);
				emitter.addText(buf);
			}
			function processSubLanguage() {
				if (modeBuffer === "") return;
				/** @type HighlightResult */
				let result = null;
				if (typeof top.subLanguage === "string") {
					if (!languages[top.subLanguage]) {
						emitter.addText(modeBuffer);
						return;
					}
					result = _highlight(top.subLanguage, modeBuffer, true, continuations[top.subLanguage]);
					continuations[top.subLanguage] = result.top;
				} else result = highlightAuto(modeBuffer, top.subLanguage.length ? top.subLanguage : null);
				if (top.relevance > 0) relevance += result.relevance;
				emitter.addSublanguage(result.emitter, result.language);
			}
			function processBuffer() {
				if (top.subLanguage != null) processSubLanguage();
				else processKeywords();
				modeBuffer = "";
			}
			/**
			* @param {Mode} mode - new mode to start
			*/
			function startNewMode(mode) {
				if (mode.className) emitter.openNode(language.classNameAliases[mode.className] || mode.className);
				top = Object.create(mode, { parent: { value: top } });
				return top;
			}
			/**
			* @param {CompiledMode } mode - the mode to potentially end
			* @param {RegExpMatchArray} match - the latest match
			* @param {string} matchPlusRemainder - match plus remainder of content
			* @returns {CompiledMode | void} - the next mode, or if void continue on in current mode
			*/
			function endOfMode(mode, match, matchPlusRemainder) {
				let matched = startsWith(mode.endRe, matchPlusRemainder);
				if (matched) {
					if (mode["on:end"]) {
						const resp = new Response(mode);
						mode["on:end"](match, resp);
						if (resp.isMatchIgnored) matched = false;
					}
					if (matched) {
						while (mode.endsParent && mode.parent) mode = mode.parent;
						return mode;
					}
				}
				if (mode.endsWithParent) return endOfMode(mode.parent, match, matchPlusRemainder);
			}
			/**
			* Handle matching but then ignoring a sequence of text
			*
			* @param {string} lexeme - string containing full match text
			*/
			function doIgnore(lexeme) {
				if (top.matcher.regexIndex === 0) {
					modeBuffer += lexeme[0];
					return 1;
				} else {
					resumeScanAtSamePosition = true;
					return 0;
				}
			}
			/**
			* Handle the start of a new potential mode match
			*
			* @param {EnhancedMatch} match - the current match
			* @returns {number} how far to advance the parse cursor
			*/
			function doBeginMatch(match) {
				const lexeme = match[0];
				const newMode = match.rule;
				const resp = new Response(newMode);
				const beforeCallbacks = [newMode.__beforeBegin, newMode["on:begin"]];
				for (const cb of beforeCallbacks) {
					if (!cb) continue;
					cb(match, resp);
					if (resp.isMatchIgnored) return doIgnore(lexeme);
				}
				if (newMode && newMode.endSameAsBegin) newMode.endRe = escape(lexeme);
				if (newMode.skip) modeBuffer += lexeme;
				else {
					if (newMode.excludeBegin) modeBuffer += lexeme;
					processBuffer();
					if (!newMode.returnBegin && !newMode.excludeBegin) modeBuffer = lexeme;
				}
				startNewMode(newMode);
				return newMode.returnBegin ? 0 : lexeme.length;
			}
			/**
			* Handle the potential end of mode
			*
			* @param {RegExpMatchArray} match - the current match
			*/
			function doEndMatch(match) {
				const lexeme = match[0];
				const matchPlusRemainder = codeToHighlight.substr(match.index);
				const endMode = endOfMode(top, match, matchPlusRemainder);
				if (!endMode) return NO_MATCH;
				const origin = top;
				if (origin.skip) modeBuffer += lexeme;
				else {
					if (!(origin.returnEnd || origin.excludeEnd)) modeBuffer += lexeme;
					processBuffer();
					if (origin.excludeEnd) modeBuffer = lexeme;
				}
				do {
					if (top.className) emitter.closeNode();
					if (!top.skip && !top.subLanguage) relevance += top.relevance;
					top = top.parent;
				} while (top !== endMode.parent);
				if (endMode.starts) {
					if (endMode.endSameAsBegin) endMode.starts.endRe = endMode.endRe;
					startNewMode(endMode.starts);
				}
				return origin.returnEnd ? 0 : lexeme.length;
			}
			function processContinuations() {
				const list = [];
				for (let current = top; current !== language; current = current.parent) if (current.className) list.unshift(current.className);
				list.forEach((item) => emitter.openNode(item));
			}
			/** @type {{type?: MatchType, index?: number, rule?: Mode}}} */
			let lastMatch = {};
			/**
			*  Process an individual match
			*
			* @param {string} textBeforeMatch - text preceeding the match (since the last match)
			* @param {EnhancedMatch} [match] - the match itself
			*/
			function processLexeme(textBeforeMatch, match) {
				const lexeme = match && match[0];
				modeBuffer += textBeforeMatch;
				if (lexeme == null) {
					processBuffer();
					return 0;
				}
				if (lastMatch.type === "begin" && match.type === "end" && lastMatch.index === match.index && lexeme === "") {
					modeBuffer += codeToHighlight.slice(match.index, match.index + 1);
					if (!SAFE_MODE) {
						/** @type {AnnotatedError} */
						const err = /* @__PURE__ */ new Error("0 width match regex");
						err.languageName = languageName;
						err.badRule = lastMatch.rule;
						throw err;
					}
					return 1;
				}
				lastMatch = match;
				if (match.type === "begin") return doBeginMatch(match);
				else if (match.type === "illegal" && !ignoreIllegals) {
					/** @type {AnnotatedError} */
					const err = /* @__PURE__ */ new Error("Illegal lexeme \"" + lexeme + "\" for mode \"" + (top.className || "<unnamed>") + "\"");
					err.mode = top;
					throw err;
				} else if (match.type === "end") {
					const processed = doEndMatch(match);
					if (processed !== NO_MATCH) return processed;
				}
				if (match.type === "illegal" && lexeme === "") return 1;
				if (iterations > 1e5 && iterations > match.index * 3) throw /* @__PURE__ */ new Error("potential infinite loop, way more iterations than matches");
				modeBuffer += lexeme;
				return lexeme.length;
			}
			const language = getLanguage(languageName);
			if (!language) {
				error(LANGUAGE_NOT_FOUND.replace("{}", languageName));
				throw new Error("Unknown language: \"" + languageName + "\"");
			}
			const md = compileLanguage(language, { plugins });
			let result = "";
			/** @type {CompiledMode} */
			let top = continuation || md;
			/** @type Record<string,CompiledMode> */
			const continuations = {};
			const emitter = new options.__emitter(options);
			processContinuations();
			let modeBuffer = "";
			let relevance = 0;
			let index = 0;
			let iterations = 0;
			let resumeScanAtSamePosition = false;
			try {
				top.matcher.considerAll();
				for (;;) {
					iterations++;
					if (resumeScanAtSamePosition) resumeScanAtSamePosition = false;
					else top.matcher.considerAll();
					top.matcher.lastIndex = index;
					const match = top.matcher.exec(codeToHighlight);
					if (!match) break;
					const processedCount = processLexeme(codeToHighlight.substring(index, match.index), match);
					index = match.index + processedCount;
				}
				processLexeme(codeToHighlight.substr(index));
				emitter.closeAllNodes();
				emitter.finalize();
				result = emitter.toHTML();
				return {
					relevance: Math.floor(relevance),
					value: result,
					language: languageName,
					illegal: false,
					emitter,
					top
				};
			} catch (err) {
				if (err.message && err.message.includes("Illegal")) return {
					illegal: true,
					illegalBy: {
						msg: err.message,
						context: codeToHighlight.slice(index - 100, index + 100),
						mode: err.mode
					},
					sofar: result,
					relevance: 0,
					value: escape$1(codeToHighlight),
					emitter
				};
				else if (SAFE_MODE) return {
					illegal: false,
					relevance: 0,
					value: escape$1(codeToHighlight),
					emitter,
					language: languageName,
					top,
					errorRaised: err
				};
				else throw err;
			}
		}
		/**
		* returns a valid highlight result, without actually doing any actual work,
		* auto highlight starts with this and it's possible for small snippets that
		* auto-detection may not find a better match
		* @param {string} code
		* @returns {HighlightResult}
		*/
		function justTextHighlightResult(code) {
			const result = {
				relevance: 0,
				emitter: new options.__emitter(options),
				value: escape$1(code),
				illegal: false,
				top: PLAINTEXT_LANGUAGE
			};
			result.emitter.addText(code);
			return result;
		}
		/**
		Highlighting with language detection. Accepts a string with the code to
		highlight. Returns an object with the following properties:
		
		- language (detected language)
		- relevance (int)
		- value (an HTML string with highlighting markup)
		- second_best (object with the same structure for second-best heuristically
		detected language, may be absent)
		
		@param {string} code
		@param {Array<string>} [languageSubset]
		@returns {AutoHighlightResult}
		*/
		function highlightAuto(code, languageSubset) {
			languageSubset = languageSubset || options.languages || Object.keys(languages);
			const plaintext = justTextHighlightResult(code);
			const results = languageSubset.filter(getLanguage).filter(autoDetection).map((name) => _highlight(name, code, false));
			results.unshift(plaintext);
			const [best, secondBest] = results.sort((a, b) => {
				if (a.relevance !== b.relevance) return b.relevance - a.relevance;
				if (a.language && b.language) {
					if (getLanguage(a.language).supersetOf === b.language) return 1;
					else if (getLanguage(b.language).supersetOf === a.language) return -1;
				}
				return 0;
			});
			/** @type {AutoHighlightResult} */
			const result = best;
			result.second_best = secondBest;
			return result;
		}
		/**
		Post-processing of the highlighted markup:
		
		- replace TABs with something more useful
		- replace real line-breaks with '<br>' for non-pre containers
		
		@param {string} html
		@returns {string}
		*/
		function fixMarkup(html) {
			if (!(options.tabReplace || options.useBR)) return html;
			return html.replace(fixMarkupRe, (match) => {
				if (match === "\n") return options.useBR ? "<br>" : match;
				else if (options.tabReplace) return match.replace(/\t/g, options.tabReplace);
				return match;
			});
		}
		/**
		* Builds new class name for block given the language name
		*
		* @param {HTMLElement} element
		* @param {string} [currentLang]
		* @param {string} [resultLang]
		*/
		function updateClassName(element, currentLang, resultLang) {
			const language = currentLang ? aliases[currentLang] : resultLang;
			element.classList.add("hljs");
			if (language) element.classList.add(language);
		}
		/** @type {HLJSPlugin} */
		const brPlugin = {
			"before:highlightElement": ({ el }) => {
				if (options.useBR) el.innerHTML = el.innerHTML.replace(/\n/g, "").replace(/<br[ /]*>/g, "\n");
			},
			"after:highlightElement": ({ result }) => {
				if (options.useBR) result.value = result.value.replace(/\n/g, "<br>");
			}
		};
		const TAB_REPLACE_RE = /^(<[^>]+>|\t)+/gm;
		/** @type {HLJSPlugin} */
		const tabReplacePlugin = { "after:highlightElement": ({ result }) => {
			if (options.tabReplace) result.value = result.value.replace(TAB_REPLACE_RE, (m) => m.replace(/\t/g, options.tabReplace));
		} };
		/**
		* Applies highlighting to a DOM node containing code. Accepts a DOM node and
		* two optional parameters for fixMarkup.
		*
		* @param {HighlightedHTMLElement} element - the HTML element to highlight
		*/
		function highlightElement(element) {
			/** @type HTMLElement */
			let node = null;
			const language = blockLanguage(element);
			if (shouldNotHighlight(language)) return;
			fire("before:highlightElement", {
				el: element,
				language
			});
			node = element;
			const text = node.textContent;
			const result = language ? highlight(text, {
				language,
				ignoreIllegals: true
			}) : highlightAuto(text);
			fire("after:highlightElement", {
				el: element,
				result,
				text
			});
			element.innerHTML = result.value;
			updateClassName(element, language, result.language);
			element.result = {
				language: result.language,
				re: result.relevance,
				relavance: result.relevance
			};
			if (result.second_best) element.second_best = {
				language: result.second_best.language,
				re: result.second_best.relevance,
				relavance: result.second_best.relevance
			};
		}
		/**
		* Updates highlight.js global options with the passed options
		*
		* @param {Partial<HLJSOptions>} userOptions
		*/
		function configure(userOptions) {
			if (userOptions.useBR) {
				deprecated("10.3.0", "'useBR' will be removed entirely in v11.0");
				deprecated("10.3.0", "Please see https://github.com/highlightjs/highlight.js/issues/2559");
			}
			options = inherit$1(options, userOptions);
		}
		/**
		* Highlights to all <pre><code> blocks on a page
		*
		* @type {Function & {called?: boolean}}
		*/
		const initHighlighting = () => {
			if (initHighlighting.called) return;
			initHighlighting.called = true;
			deprecated("10.6.0", "initHighlighting() is deprecated.  Use highlightAll() instead.");
			document.querySelectorAll("pre code").forEach(highlightElement);
		};
		function initHighlightingOnLoad() {
			deprecated("10.6.0", "initHighlightingOnLoad() is deprecated.  Use highlightAll() instead.");
			wantsHighlight = true;
		}
		let wantsHighlight = false;
		/**
		* auto-highlights all pre>code elements on the page
		*/
		function highlightAll() {
			if (document.readyState === "loading") {
				wantsHighlight = true;
				return;
			}
			document.querySelectorAll("pre code").forEach(highlightElement);
		}
		function boot() {
			if (wantsHighlight) highlightAll();
		}
		if (typeof window !== "undefined" && window.addEventListener) window.addEventListener("DOMContentLoaded", boot, false);
		/**
		* Register a language grammar module
		*
		* @param {string} languageName
		* @param {LanguageFn} languageDefinition
		*/
		function registerLanguage(languageName, languageDefinition) {
			let lang = null;
			try {
				lang = languageDefinition(hljs);
			} catch (error$1) {
				error("Language definition for '{}' could not be registered.".replace("{}", languageName));
				if (!SAFE_MODE) throw error$1;
				else error(error$1);
				lang = PLAINTEXT_LANGUAGE;
			}
			if (!lang.name) lang.name = languageName;
			languages[languageName] = lang;
			lang.rawDefinition = languageDefinition.bind(null, hljs);
			if (lang.aliases) registerAliases(lang.aliases, { languageName });
		}
		/**
		* Remove a language grammar module
		*
		* @param {string} languageName
		*/
		function unregisterLanguage(languageName) {
			delete languages[languageName];
			for (const alias of Object.keys(aliases)) if (aliases[alias] === languageName) delete aliases[alias];
		}
		/**
		* @returns {string[]} List of language internal names
		*/
		function listLanguages() {
			return Object.keys(languages);
		}
		/**
		intended usage: When one language truly requires another
		
		Unlike `getLanguage`, this will throw when the requested language
		is not available.
		
		@param {string} name - name of the language to fetch/require
		@returns {Language | never}
		*/
		function requireLanguage(name) {
			deprecated("10.4.0", "requireLanguage will be removed entirely in v11.");
			deprecated("10.4.0", "Please see https://github.com/highlightjs/highlight.js/pull/2844");
			const lang = getLanguage(name);
			if (lang) return lang;
			throw new Error("The '{}' language is required, but not loaded.".replace("{}", name));
		}
		/**
		* @param {string} name - name of the language to retrieve
		* @returns {Language | undefined}
		*/
		function getLanguage(name) {
			name = (name || "").toLowerCase();
			return languages[name] || languages[aliases[name]];
		}
		/**
		*
		* @param {string|string[]} aliasList - single alias or list of aliases
		* @param {{languageName: string}} opts
		*/
		function registerAliases(aliasList, { languageName }) {
			if (typeof aliasList === "string") aliasList = [aliasList];
			aliasList.forEach((alias) => {
				aliases[alias.toLowerCase()] = languageName;
			});
		}
		/**
		* Determines if a given language has auto-detection enabled
		* @param {string} name - name of the language
		*/
		function autoDetection(name) {
			const lang = getLanguage(name);
			return lang && !lang.disableAutodetect;
		}
		/**
		* Upgrades the old highlightBlock plugins to the new
		* highlightElement API
		* @param {HLJSPlugin} plugin
		*/
		function upgradePluginAPI(plugin) {
			if (plugin["before:highlightBlock"] && !plugin["before:highlightElement"]) plugin["before:highlightElement"] = (data) => {
				plugin["before:highlightBlock"](Object.assign({ block: data.el }, data));
			};
			if (plugin["after:highlightBlock"] && !plugin["after:highlightElement"]) plugin["after:highlightElement"] = (data) => {
				plugin["after:highlightBlock"](Object.assign({ block: data.el }, data));
			};
		}
		/**
		* @param {HLJSPlugin} plugin
		*/
		function addPlugin(plugin) {
			upgradePluginAPI(plugin);
			plugins.push(plugin);
		}
		/**
		*
		* @param {PluginEvent} event
		* @param {any} args
		*/
		function fire(event, args) {
			const cb = event;
			plugins.forEach(function(plugin) {
				if (plugin[cb]) plugin[cb](args);
			});
		}
		/**
		Note: fixMarkup is deprecated and will be removed entirely in v11
		
		@param {string} arg
		@returns {string}
		*/
		function deprecateFixMarkup(arg) {
			deprecated("10.2.0", "fixMarkup will be removed entirely in v11.0");
			deprecated("10.2.0", "Please see https://github.com/highlightjs/highlight.js/issues/2534");
			return fixMarkup(arg);
		}
		/**
		*
		* @param {HighlightedHTMLElement} el
		*/
		function deprecateHighlightBlock(el) {
			deprecated("10.7.0", "highlightBlock will be removed entirely in v12.0");
			deprecated("10.7.0", "Please use highlightElement now.");
			return highlightElement(el);
		}
		Object.assign(hljs, {
			highlight,
			highlightAuto,
			highlightAll,
			fixMarkup: deprecateFixMarkup,
			highlightElement,
			highlightBlock: deprecateHighlightBlock,
			configure,
			initHighlighting,
			initHighlightingOnLoad,
			registerLanguage,
			unregisterLanguage,
			listLanguages,
			getLanguage,
			registerAliases,
			requireLanguage,
			autoDetection,
			inherit: inherit$1,
			addPlugin,
			vuePlugin: BuildVuePlugin(hljs).VuePlugin
		});
		hljs.debugMode = function() {
			SAFE_MODE = false;
		};
		hljs.safeMode = function() {
			SAFE_MODE = true;
		};
		hljs.versionString = version;
		for (const key in MODES) if (typeof MODES[key] === "object") deepFreezeEs6(MODES[key]);
		Object.assign(hljs, MODES);
		hljs.addPlugin(brPlugin);
		hljs.addPlugin(mergeHTMLPlugin);
		hljs.addPlugin(tabReplacePlugin);
		return hljs;
	};
	module.exports = HLJS({});
}));
//#endregion
//#region node_modules/format/format.js
var require_format = /* @__PURE__ */ __commonJSMin(((exports, module) => {
	(function() {
		var namespace;
		if (typeof module !== "undefined") namespace = module.exports = format;
		else namespace = function() {
			return this || (0, eval)("this");
		}();
		namespace.format = format;
		namespace.vsprintf = vsprintf;
		if (typeof console !== "undefined" && typeof console.log === "function") namespace.printf = printf;
		function printf() {
			console.log(format.apply(null, arguments));
		}
		function vsprintf(fmt, replacements) {
			return format.apply(null, [fmt].concat(replacements));
		}
		function format(fmt) {
			var argIndex = 1, args = [].slice.call(arguments), i = 0, n = fmt.length, result = "", c, escaped = false, arg, tmp, leadingZero = false, precision, nextArg = function() {
				return args[argIndex++];
			}, slurpNumber = function() {
				var digits = "";
				while (/\d/.test(fmt[i])) {
					digits += fmt[i++];
					c = fmt[i];
				}
				return digits.length > 0 ? parseInt(digits) : null;
			};
			for (; i < n; ++i) {
				c = fmt[i];
				if (escaped) {
					escaped = false;
					if (c == ".") {
						leadingZero = false;
						c = fmt[++i];
					} else if (c == "0" && fmt[i + 1] == ".") {
						leadingZero = true;
						i += 2;
						c = fmt[i];
					} else leadingZero = true;
					precision = slurpNumber();
					switch (c) {
						case "b":
							result += parseInt(nextArg(), 10).toString(2);
							break;
						case "c":
							arg = nextArg();
							if (typeof arg === "string" || arg instanceof String) result += arg;
							else result += String.fromCharCode(parseInt(arg, 10));
							break;
						case "d":
							result += parseInt(nextArg(), 10);
							break;
						case "f":
							tmp = String(parseFloat(nextArg()).toFixed(precision || 6));
							result += leadingZero ? tmp : tmp.replace(/^0/, "");
							break;
						case "j":
							result += JSON.stringify(nextArg());
							break;
						case "o":
							result += "0" + parseInt(nextArg(), 10).toString(8);
							break;
						case "s":
							result += nextArg();
							break;
						case "x":
							result += "0x" + parseInt(nextArg(), 10).toString(16);
							break;
						case "X":
							result += "0x" + parseInt(nextArg(), 10).toString(16).toUpperCase();
							break;
						default:
							result += c;
							break;
					}
				} else if (c === "%") escaped = true;
				else result += c;
			}
			return result;
		}
	})();
}));
//#endregion
//#region node_modules/fault/index.js
var require_fault = /* @__PURE__ */ __commonJSMin(((exports, module) => {
	var formatter = require_format();
	var fault = create(Error);
	module.exports = fault;
	fault.eval = create(EvalError);
	fault.range = create(RangeError);
	fault.reference = create(ReferenceError);
	fault.syntax = create(SyntaxError);
	fault.type = create(TypeError);
	fault.uri = create(URIError);
	fault.create = create;
	function create(EConstructor) {
		FormattedError.displayName = EConstructor.displayName || EConstructor.name;
		return FormattedError;
		function FormattedError(format) {
			if (format) format = formatter.apply(null, arguments);
			return new EConstructor(format);
		}
	}
}));
//#endregion
//#region node_modules/lowlight/lib/core.js
var require_core = /* @__PURE__ */ __commonJSMin(((exports) => {
	var high = require_core$1();
	var fault = require_fault();
	exports.highlight = highlight;
	exports.highlightAuto = highlightAuto;
	exports.registerLanguage = registerLanguage;
	exports.listLanguages = listLanguages;
	exports.registerAlias = registerAlias;
	Emitter.prototype.addText = text;
	Emitter.prototype.addKeyword = addKeyword;
	Emitter.prototype.addSublanguage = addSublanguage;
	Emitter.prototype.openNode = open;
	Emitter.prototype.closeNode = close;
	Emitter.prototype.closeAllNodes = noop;
	Emitter.prototype.finalize = noop;
	Emitter.prototype.toHTML = toHtmlNoop;
	var defaultPrefix = "hljs-";
	function highlight(name, value, options) {
		var before = high.configure({});
		var prefix = (options || {}).prefix;
		var result;
		if (typeof name !== "string") throw fault("Expected `string` for name, got `%s`", name);
		if (!high.getLanguage(name)) throw fault("Unknown language: `%s` is not registered", name);
		if (typeof value !== "string") throw fault("Expected `string` for value, got `%s`", value);
		if (prefix === null || prefix === void 0) prefix = defaultPrefix;
		high.configure({
			__emitter: Emitter,
			classPrefix: prefix
		});
		result = high.highlight(value, {
			language: name,
			ignoreIllegals: true
		});
		high.configure(before || {});
		/* istanbul ignore if - Highlight.js seems to use this (currently) for broken
		* grammars, so let’s keep it in there just to be sure. */
		if (result.errorRaised) throw result.errorRaised;
		return {
			relevance: result.relevance,
			language: result.language,
			value: result.emitter.rootNode.children
		};
	}
	function highlightAuto(value, options) {
		var settings = options || {};
		var subset = settings.subset || high.listLanguages();
		var prefix = settings.prefix;
		var length = subset.length;
		var index = -1;
		var result;
		var secondBest;
		var current;
		var name;
		if (prefix === null || prefix === void 0) prefix = defaultPrefix;
		if (typeof value !== "string") throw fault("Expected `string` for value, got `%s`", value);
		secondBest = {
			relevance: 0,
			language: null,
			value: []
		};
		result = {
			relevance: 0,
			language: null,
			value: []
		};
		while (++index < length) {
			name = subset[index];
			if (!high.getLanguage(name)) continue;
			current = highlight(name, value, options);
			current.language = name;
			if (current.relevance > secondBest.relevance) secondBest = current;
			if (current.relevance > result.relevance) {
				secondBest = result;
				result = current;
			}
		}
		if (secondBest.language) result.secondBest = secondBest;
		return result;
	}
	function registerLanguage(name, syntax) {
		high.registerLanguage(name, syntax);
	}
	function listLanguages() {
		return high.listLanguages();
	}
	function registerAlias(name, alias) {
		var map = name;
		var key;
		if (alias) {
			map = {};
			map[name] = alias;
		}
		for (key in map) high.registerAliases(map[key], { languageName: key });
	}
	function Emitter(options) {
		this.options = options;
		this.rootNode = { children: [] };
		this.stack = [this.rootNode];
	}
	function addKeyword(value, name) {
		this.openNode(name);
		this.addText(value);
		this.closeNode();
	}
	function addSublanguage(other, name) {
		var stack = this.stack;
		var current = stack[stack.length - 1];
		var results = other.rootNode.children;
		var node = name ? {
			type: "element",
			tagName: "span",
			properties: { className: [name] },
			children: results
		} : results;
		current.children = current.children.concat(node);
	}
	function text(value) {
		var stack = this.stack;
		var current;
		var tail;
		if (value === "") return;
		current = stack[stack.length - 1];
		tail = current.children[current.children.length - 1];
		if (tail && tail.type === "text") tail.value += value;
		else current.children.push({
			type: "text",
			value
		});
	}
	function open(name) {
		var stack = this.stack;
		var className = this.options.classPrefix + name;
		var current = stack[stack.length - 1];
		var child = {
			type: "element",
			tagName: "span",
			properties: { className: [className] },
			children: []
		};
		current.children.push(child);
		stack.push(child);
	}
	function close() {
		this.stack.pop();
	}
	function toHtmlNoop() {
		return "";
	}
	function noop() {}
}));
//#endregion
export { require_core as t };

//# sourceMappingURL=core-DZZEJm4g.js.map