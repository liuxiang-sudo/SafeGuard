// Exercise the shared script on pages with missing form fields, without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../safeguard/web/static/js/script.js'), 'utf8');

function exercise({ withNavigation = false, withForm = false } = {}) {
    const documentEvents = {};
    const formEvents = {};
    const anchorEvents = {};
    const navigation = { value: '' };
    const anchor = {
        getAttribute: () => '#',
        addEventListener: (name, callback) => { anchorEvents[name] = callback; }
    };
    const elements = {};
    if (withNavigation) elements.navigation = navigation;
    if (withForm) elements['detection-form'] = {
        addEventListener: (name, callback) => { formEvents[name] = callback; }
    };
    const context = {
        console, Date, Number, Object,
        document: {
            addEventListener: (name, callback) => { documentEvents[name] = callback; },
            getElementById: id => elements[id] || null,
            querySelectorAll: selector => selector === 'a[href^="#"]' ? [anchor] : [],
            referrer: ''
        },
        window: { addEventListener() {}, innerWidth: 1200, innerHeight: 800 },
        navigator: { userAgent: 'test-browser' },
        sessionStorage: {
            getItem() { throw new Error('storage blocked'); },
            setItem() { throw new Error('storage blocked'); }
        },
        setTimeout: callback => callback()
    };
    vm.runInNewContext(source, context);
    documentEvents.DOMContentLoaded();
    documentEvents.click({ clientX: 0, clientY: 0 });
    anchorEvents.click.call(anchor, { preventDefault() {} });
    if (withForm) formEvents.submit({});
    if (withNavigation) assert.equal(navigation.value, '0');
}

exercise();
exercise({ withNavigation: true });
exercise({ withForm: true });
console.log('Browser script: missing fields, blocked storage, placeholder links and partial forms passed.');
